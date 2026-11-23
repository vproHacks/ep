# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "plotly",
# ]
# ///
"""Electrophysiology Data, End to End — a marimo notebook.

Edit locally:   marimo edit ep_primer.py
Run locally:    marimo run ep_primer.py
Publish:        marimo export html-wasm ep_primer.py -o _site --mode run --show-code

The notebook reads one precomputed file, `public/ep_primer_data.npz`, built by
`scripts/build_bundle.py` from the OpenEP MATLAB exports. That indirection is
what lets the same file run on a laptop and in a browser.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Electrophysiology Data, End to End",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _(mo):
    # The published page runs in the browser under Pyodide, where `epkit` is not
    # on the path until we install the wheel that ships next to the notebook.
    try:
        import epkit
    except ModuleNotFoundError:
        import micropip

        await micropip.install(
            str(mo.notebook_location() / "public" / "epkit-0.1.0-py3-none-any.whl")
        )
        import epkit

    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    from scipy import spatial, stats
    from scipy import signal as sps

    np.set_printoptions(precision=3, suppress=True)
    return epkit, go, np, plt, spatial, sps, stats


@app.cell(hide_code=True)
def _(mo):
    def plain(text):
        """A plain-English restatement, for readers with no clinical background."""
        return mo.callout(mo.md(f"**In plain English.** {text}"), kind="info")

    def takeaway(text):
        return mo.callout(mo.md(f"**Worth keeping.** {text}"), kind="success")

    def trap(text):
        return mo.callout(mo.md(f"**Trap.** {text}"), kind="warn")

    def deeper(title, text):
        return mo.accordion({f"Going deeper — {title}": mo.md(text)})

    def jargon(pairs):
        rows = "\n".join(f"| **{term}** | {meaning} |" for term, meaning in pairs)
        return mo.accordion(
            {
                "Words used in this section": mo.md(
                    f"| term | what it means |\n|:---|:---|\n{rows}"
                )
            }
        )

    return deeper, jargon, plain, takeaway, trap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electrophysiology Data, End to End

    ### A working introduction to the data behind cardiac ablation — for people who have never seen any of it before

    The heart runs on electricity. When that electricity goes wrong, a cardiologist
    can thread a wire into the heart, record the electrical activity from the inside,
    draw a coloured 3D map of it, and burn away the tissue causing the problem.

    This notebook is about **the data that procedure produces**: what it physically
    is, how the coloured maps get made, and — the part nobody puts on the screen —
    how much of those maps is measurement and how much is inference.

    Everything here runs on **real clinical recordings**: a CARTO export from a
    catheter ablation, distributed with the [OpenEP](https://openep.io) project.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Do I need a medical background? (No — here is what you do need)": mo.md(
                r"""
    **No clinical knowledge is assumed.** Every anatomical or clinical term is
    explained where it first appears, and there is a full glossary at the end.

    What helps: comfort reading numpy, and a willingness to look at a plot and ask
    *how would I know if this were wrong?* That question is the whole subject.

    **Two ways to read this.**

    - *Prose only.* Skip every code cell. The narrative stands on its own, and the
      numbers in it are interpolated straight from the computation beside them,
      so the text cannot quietly drift from the data.
    - *Prose plus code.* The code is the argument. Every claim in the text is
      something the cell above or below it actually computed.

    **Three labels appear throughout**, and they matter more than they look:

    | label | meaning |
    |:---|:---|
    | **MEASURED** | a number that came off the recording system |
    | **DERIVED** | a number this notebook computed from measured data |
    | **ILLUSTRATIVE** | a demonstration of a method the data cannot fully support |

    Nearly everything a clinician looks at on screen is *derived*. Keeping the
    distinction visible is the point of the exercise.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 0 — Orientation: the heart as an electrical machine

    Skip this part if you already know what a pulmonary vein isolation is. If you
    don't, it is fifteen paragraphs that will make the other ten parts legible.
    """)
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### The heart is a pump with its own wiring

    Four chambers: two small ones on top (the **atria**), two large ones underneath
    (the **ventricles**). Blood arrives in the atria, gets nudged down into the
    ventricles, and the ventricles squeeze it out to the lungs and the body.

    The squeeze is not commanded by the brain. Heart muscle cells contract when
    their electrical state flips, and one cell flipping drags its neighbours with
    it. So a single spark in the right place sets off a **wave of electrical
    activation** that sweeps across the muscle, and the muscle contracts in the
    order the wave reaches it. The wiring is not wires; it is the muscle itself.

    Roughly once a second, a small patch of tissue at the top of the right atrium
    fires spontaneously, the wave spreads across both atria, funnels through a
    single electrical gateway into the ventricles, and sweeps through them. That
    ordered sequence is one heartbeat.
    """
            ),
            plain(
                "The heart's muscle carries its own electrical wave, and the "
                "contraction follows the wave. Everything in this notebook is a "
                "measurement of that wave."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(deeper, mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### What goes wrong

    An **arrhythmia** is that wave misbehaving. Two families cover most of what
    gets treated in a catheter lab:

    - **Focal.** Somewhere in the muscle, a patch of tissue starts firing on its
      own, faster than the normal pacemaker, and hijacks the rhythm. The wave
      spreads outward from that spot like a stone dropped in a pond. Cure the
      arrhythmia by destroying that one spot.
    - **Reentrant.** The wave fails to die out. Normally a wave sweeps across the
      muscle and extinguishes, because everywhere it could go next is still
      *refractory* — recently activated tissue that needs a moment before it can
      fire again. But if the wave can travel a long enough or slow enough path, it
      arrives back where it started to find tissue that has recovered, and
      re-excites it. Now it circulates indefinitely: a self-sustaining loop, going
      round at two or three times the normal heart rate. Cure it by cutting the
      loop.

    Diseased or scarred tissue conducts slowly, and slow conduction is what makes
    reentry possible — it gives the tissue ahead of the wave time to recover. This
    is why the *slow-conduction* regions mapped in Part 7 matter clinically: they
    are where circuits live.

    ### Atrial fibrillation, the reason for this particular dataset

    The recordings here come from a **left atrium**. The commonest reason to map one
    is **atrial fibrillation (AF)** — the atria stop beating in an organised way and
    quiver instead, at several hundred disorganised activations a minute. Blood
    stagnates, clots form, and strokes follow; the patient feels awful.

    The single most important observation in the field, from 1998, is that most AF
    is triggered by rapid firing from inside the **pulmonary veins** — the four
    vessels returning oxygenated blood from the lungs into the left atrium. Muscle
    sleeves extend a short way up into those veins and fire spontaneously.

    That gives the standard operation: **pulmonary vein isolation**. Don't chase
    every trigger. Instead burn a continuous ring of scar around the veins where
    they join the atrium, so that whatever fires inside them can no longer get out.
    The 90 burn sites in this dataset are that ring.
    """
            ),
            plain(
                "The commonest heart-rhythm operation draws a scar ring around the "
                "four veins entering the upper-left chamber, to fence in the tissue "
                "that starts the arrhythmia. This dataset is one such operation."
            ),
            deeper(
                "why scar is the treatment, and why that is safe",
                r"""
    Radiofrequency ablation heats a spot of tissue to around 50 °C. The muscle cells
    there die and are replaced by fibrous scar, which conducts no electricity at all.
    The heart wall is a few millimetres thick and the lesions are a few millimetres
    across, so the pump loses a negligible amount of muscle; what it loses is the
    *electrical* connection through that spot. The therapy is not removing tissue —
    it is removing a conduction path.

    The same logic drives all the mapping in this notebook. You are not looking for
    disease in the way a radiologist looks for a tumour. You are looking for the
    route the wave takes, so you can decide where to interrupt it.
    """,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(jargon, mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### What actually happens in the room

    The patient is on a table, awake or lightly sedated. Nothing is opened. Long
    thin tubes — **catheters** — go in through a vein at the top of the leg and are
    pushed up to the heart, with X-ray and a magnetic tracking system showing where
    they are. The left atrium sits behind the right, so getting there means a
    deliberate small puncture through the wall between them: the **transseptal**
    step.

    Each catheter carries **electrodes** — small metal rings a couple of millimetres
    apart near its tip. An electrode in the bloodstream can measure the electrical
    field produced by nearby muscle, and can also pass current *into* the tissue,
    which is how the burning is done.

    From there, the procedure is a data-collection exercise:

    1. **Build the shell.** Sweep a catheter around the inside of the chamber. The
       system remembers the outermost places the catheter reached and stitches them
       into a 3D surface. That surface is the coloured heart on the screen. It is
       not a scan — it is a record of where the catheter has been.
    2. **Collect points.** At hundreds of positions on that surface, record the
       local electrical signal and the moment the wave arrived.
    3. **Draw the maps.** Colour the surface by arrival time, or by signal size, and
       look at the picture.
    4. **Decide.** Focal or reentrant? Where is the target?
    5. **Burn.** Deliver lesions, one at a time, each recorded as a marker.
    6. **Check.** Re-map to prove the target is electrically silent.

    Everything in the exported file is the residue of those six steps, and Parts 3
    to 8 walk through them in the same order.
    """
            ),
            plain(
                "Wires in through a leg vein, a map built by touching the inside of "
                "the chamber, a diagnosis read off the colours, then a line of small "
                "burns. The file we are reading is what that leaves behind."
            ),
            jargon(
                [
                    ("atrium", "one of the two small upper chambers of the heart"),
                    (
                        "left atrium",
                        "the upper-left chamber, where the four lung veins arrive; the chamber mapped here",
                    ),
                    (
                        "pulmonary vein",
                        "one of four vessels bringing blood from the lungs into the left atrium",
                    ),
                    (
                        "mitral valve",
                        "the one-way door from the left atrium down into the left ventricle",
                    ),
                    (
                        "appendage",
                        "a small blind pouch off the left atrium, a leftover of embryology",
                    ),
                    ("catheter", "a steerable tube threaded through a vein into the heart"),
                    (
                        "electrode",
                        "a metal contact on a catheter that records voltage, or delivers burning current",
                    ),
                    (
                        "ablation",
                        "deliberately destroying a small patch of tissue, here by heating it",
                    ),
                    (
                        "electrophysiology (EP)",
                        "the study and treatment of the heart's electrical behaviour",
                    ),
                    (
                        "CARTO",
                        "one of the commercial mapping systems; this dataset is a CARTO export",
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The chamber, schematically

    Left atrium seen from behind, which is how it is usually drawn. Four veins
    entering, one valve leaving, one pouch on the side. The mesh loaded below is
    this shape, measured in a real patient.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
    <svg viewBox="0 0 640 340" width="100%" style="max-width:640px;display:block;margin:auto"
         xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Schematic of the left atrium with four pulmonary veins, the appendage and the mitral valve">
      <style>
        .wall  { fill: #f0dede; stroke: #b07f7f; stroke-width: 2.5; }
        .vein  { fill: #dfe8f5; stroke: #6f89ad; stroke-width: 2.5; }
        .ring  { fill: none; stroke: #c0392b; stroke-width: 4; stroke-dasharray: 7 5; }
        .lbl   { font: 500 13px system-ui, sans-serif; fill: #4a4a4a; }
        .lblk  { font: 600 13px system-ui, sans-serif; fill: #8b3a3a; }
        .lead  { stroke: #9a9a9a; stroke-width: 1; }
        @media (prefers-color-scheme: dark) {
          .wall { fill: #4a3a3a; stroke: #c99; }
          .vein { fill: #33445c; stroke: #8fa8cc; }
          .lbl  { fill: #cfcfcf; } .lblk { fill: #e6a8a8; } .lead { stroke: #777; }
        }
      </style>
      <ellipse class="wall" cx="320" cy="180" rx="140" ry="105"/>
      <rect class="vein" x="150" y="40"  width="46" height="70" rx="20"/>
      <rect class="vein" x="150" y="230" width="46" height="70" rx="20"/>
      <rect class="vein" x="444" y="40"  width="46" height="70" rx="20"/>
      <rect class="vein" x="444" y="230" width="46" height="70" rx="20"/>
      <ellipse class="vein" cx="530" cy="170" rx="34" ry="20" transform="rotate(-20 530 170)"/>
      <ellipse class="wall" cx="320" cy="292" rx="52" ry="22"/>
      <ellipse class="ring" cx="196" cy="140" rx="72" ry="86"/>
      <ellipse class="ring" cx="444" cy="140" rx="72" ry="86"/>
      <line class="lead" x1="120" y1="30"  x2="160" y2="55"/>
      <line class="lead" x1="120" y1="310" x2="160" y2="285"/>
      <line class="lead" x1="520" y1="30"  x2="480" y2="55"/>
      <line class="lead" x1="520" y1="310" x2="480" y2="285"/>
      <text class="lbl" x="20"  y="26">left superior pulmonary vein</text>
      <text class="lbl" x="20"  y="326">left inferior pulmonary vein</text>
      <text class="lbl" x="524" y="26">right superior</text>
      <text class="lbl" x="524" y="326">right inferior</text>
      <text class="lbl" x="556" y="150">appendage</text>
      <text class="lbl" x="264" y="330">mitral valve</text>
      <line class="lead" x1="96" y1="182" x2="126" y2="170"/>
      <line class="lead" x1="548" y1="248" x2="512" y2="212"/>
      <text class="lblk" x="12" y="186">ablation ring</text>
      <text class="lblk" x="632" y="252" text-anchor="end">ablation ring</text>
    </svg>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, takeaway):
    mo.vstack(
        [
            mo.md(
                r"""
    Two features of that picture come back repeatedly:

    - The **veins and the valve are holes**. The chamber is not a closed bag; it is
      a bag with six openings. In Part 3 we find those openings in the data using
      nothing but the connectivity of the triangles, and they land exactly where
      anatomy says they should.
    - The **dashed rings are the therapy**. In Part 8 the 90 recorded burn sites
      are plotted on the real geometry, and they trace those two rings.
    """
            ),
            takeaway(
                "Anatomy here is not decoration. The holes in the mesh, the position "
                "of the burns, and the shape of the activation wave are all things "
                "you can read straight out of the arrays."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The file itself

    An EP study export is three things bolted together:

    | | what it is | how big, here |
    |:---|:---|:---|
    | **geometry** | a triangulated 3D surface — the shell | thousands of vertices |
    | **electrical points** | recordings made at positions on that surface | hundreds to thousands |
    | **ablation record** | where therapy was delivered, and how | one row per lesion |

    Everything else on an EP screen is derived from those three. Let's open it.
    """)
    return


@app.cell
def _(epkit, mo):
    # MEASURED. One precomputed bundle, built offline from the OpenEP MATLAB
    # export by scripts/build_bundle.py. See the appendix for exactly what was
    # done to it — the short version is: nothing but repacking.
    bundle = epkit.load_bundle(mo.notebook_location() / "public" / "ep_primer_data.npz")

    print(bundle)
    print(bundle.meta["provenance"])
    return (bundle,)


@app.cell
def _(bundle, np):
    # MEASURED. Unpack the arrays the whole notebook works from.
    points = bundle.points.astype(float)             # (n_vertices, 3) mm
    triangles = bundle.triangles                     # (n_triangles, 3) vertex indices

    bipolar = bundle.bipolar_egm.astype(float)       # (n_points, n_samples) mV
    unipolar = bundle.unipolar_egm.astype(float)     # (n_points, n_samples, 2) mV
    ecg_signal = bundle.ecg.astype(float)            # (n_points, n_samples) mV

    egm_points = bundle.egm_points.astype(float)     # where each recording was made
    surface_points = bundle.surface_points.astype(float)   # projected onto the shell

    fs = bundle.meta["sampling_frequency_hz"]
    n_samples = bundle.meta["n_samples"]
    n_points = bundle.meta["n_points"]
    times_ms = np.arange(n_samples) / fs * 1000.0

    summary = [
        ("Geometry: vertices", f"{len(points):,}"),
        ("Geometry: triangles", f"{len(triangles):,}"),
        ("Mapping points", f"{n_points:,}"),
        ("Samples per recording", f"{n_samples:,}"),
        ("Sampling frequency", f"{fs:.0f} Hz"),
        ("Duration of each recording", f"{n_samples / fs:.2f} s"),
        ("Bipolar recording array", str(bipolar.shape)),
        ("Unipolar recording array", str(unipolar.shape)),
        ("ECG array", str(ecg_signal.shape)),
        ("Ablation telemetry samples", f"{len(bundle.ablation_times):,}"),
        ("Ablation lesion tags", f"{len(bundle.tag_points):,}"),
    ]
    _width = max(len(label) for label, _ in summary)
    for _label, _value in summary:
        print(f"{_label:<{_width}} : {_value}")
    return (
        bipolar,
        egm_points,
        fs,
        n_points,
        n_samples,
        points,
        surface_points,
        times_ms,
        triangles,
        unipolar,
    )


@app.cell(hide_code=True)
def _(mo, n_points, points, takeaway, triangles):
    mo.vstack(
        [
            mo.md(
                rf"""
    ### The shape of the problem, in one ratio

    Look at the two numbers that matter most: **{n_points:,} electrical
    recordings**, and **{len(points):,} geometry vertices** to colour.

    Every coloured map you have ever seen in an EP lab — or in a paper, or on a
    conference slide — is an interpolation from the first onto the second. Roughly
    **{len(points) / n_points:.0f} vertices are coloured for every one that was
    actually measured**, and the triangles between them
    ({len(triangles):,} of them) are shaded by the graphics card.

    Part 4 takes that ratio seriously, and it is the most important part of the
    notebook.
    """
            ),
            takeaway(
                f"About {100 * (1 - n_points / len(points)):.0f}% of the coloured "
                "surface is inference. That is not a flaw in this dataset; it is "
                "how every clinical electroanatomic map is made."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — What a recording physically is

    Before any map, the raw material: a wiggly line. Where does it come from, and
    what is it a measurement *of*? This part answers that from first principles,
    because every later interpretation depends on it.
    """)
    return


@app.cell(hide_code=True)
def _(deeper, jargon, mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### One cell

    A heart muscle cell sits at about **−85 mV** across its membrane when resting:
    the inside is negative relative to the outside, held there by pumps that keep
    sodium, potassium and calcium at different concentrations on the two sides.

    Studded through the membrane are **ion channels** — pores that open and close,
    and crucially, whose opening depends on the very voltage they control.

    That feedback is the whole story. Push the membrane voltage up past about
    −65 mV and sodium channels open, sodium rushes in, which pushes the voltage up
    further, which opens more channels. Within a millisecond the cell has flipped
    to about +20 mV. That flip is the **action potential**. Calcium channels then
    hold it up there for a couple of hundred milliseconds — the plateau that makes
    a heart cell's action potential last ~250 ms where a nerve cell's lasts ~1 ms —
    and finally potassium channels let the voltage fall back down.

    While the cell is depolarised and for a short while afterwards, it *cannot* be
    re-triggered. That interval is the **refractory period**, and it is the reason
    a heart does not simply buzz.
    """
            ),
            plain(
                "Each muscle cell is a tiny switch. Nudge it past a threshold and it "
                "flips itself the rest of the way, stays flipped for about a quarter "
                "of a second, then resets — and it ignores anything that arrives "
                "while it is flipped."
            ),
            deeper(
                "the same thing said in engineering",
                r"""
    A myocyte is a relaxation oscillator with a long refractory period, coupled
    resistively to its neighbours. The tissue is therefore an **excitable medium**:
    a bistable-ish reaction–diffusion system supporting travelling waves.

    That framing predicts the clinically important behaviours directly. Waves
    annihilate on collision (both sides refractory). Waves can circulate around an
    unexcitable obstacle if the loop length exceeds wavelength = conduction velocity
    × refractory period — which is exactly the reentry condition, and exactly why
    both *slowing conduction* (disease, drugs) and *shortening refractoriness*
    (catecholamines) make arrhythmia easier. Spiral waves and their breakup are the
    textbook route to fibrillation.
    """,
            ),
            jargon(
                [
                    ("myocyte", "a muscle cell"),
                    ("depolarise", "the inside of the cell becoming less negative — the switch flipping"),
                    ("repolarise", "the cell resetting to its negative resting state"),
                    ("refractory", "temporarily unable to be re-activated"),
                    ("action potential", "the whole flip-and-reset voltage excursion of one cell"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Many cells: a wave

    Muscle cells are electrically joined to their neighbours through **gap
    junctions** — protein channels that let ions pass directly from one cell into
    the next. So a cell that flips drags its neighbours over threshold, and they
    drag theirs.

    The result is a **wavefront**: a moving boundary between tissue that has already
    flipped and tissue that has not, sweeping across the muscle at roughly
    **0.3–1.0 metres per second**. In the units used throughout this notebook that
    is 300–1000 mm/s, or about 1 mm every millisecond.

    Behind the wavefront trails the refractory tail. Ahead of it, resting tissue.
    That is the entire physical system this notebook measures.
    """
            ),
            plain(
                "Think of a crowd doing a stadium wave. Each person stands because "
                "their neighbour just did. Nobody is coordinating it; the pattern "
                "propagates. And a person who just stood up can't stand up again "
                "immediately — which is why the wave keeps moving forward instead of "
                "washing back."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plain, trap):
    mo.vstack(
        [
            mo.md(
                r"""
    ### The measurement, and the single most-misunderstood point in the subject

    Here is what an electrode in the chamber does **not** do: it does not measure
    the action potential of a cell. Nothing in this dataset is a transmembrane
    voltage.

    What it measures is this. A wavefront is a moving boundary between depolarised
    and resting tissue — that is, a moving *separation of charge*, and a charge
    separation produces an electric field in the blood and tissue around it. An
    electrode sitting in the blood a millimetre or two away samples that field,
    against some reference. The number recorded is a few millivolts of
    **extracellular potential**, produced by many thousands of cells flipping in
    concert.

    Three consequences, and they are the basis of every clinical reading:

    - **Fast, synchronised depolarisation of healthy muscle right under the
      electrode** → a large, sharp deflection.
    - **Slow or disorganised depolarisation** — surviving strands of muscle
      threading through scar, each flipping at a slightly different moment →
      a small, smeared, multi-humped signal.
    - **Dead scar under the electrode** → almost nothing local; whatever is left is
      *far-field*, the tail of the field from healthy tissue somewhere else.

    So the size of the signal is a proxy for how much healthy muscle is under the
    electrode, and the timing of the signal is a proxy for when the wave arrived
    there. Those two proxies are the whole of clinical mapping: **voltage maps** and
    **activation maps**.
    """
            ),
            plain(
                "The catheter is a microphone in a room, not a probe in a cell. It "
                "hears the sound of muscle activating nearby — loud and crisp if the "
                "muscle is healthy and close, muffled and distant if it is not."
            ),
            trap(
                "Because it is a field measurement, an electrode always hears "
                "*something*, including from tissue centimetres away. A signal is "
                "not proof of local muscle. Distinguishing local from far-field is "
                "the job of the bipolar recording in Part 2."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A toy model, to see where the wiggle comes from

    Rather than assert the shape of an electrogram, let's generate one. The model
    below is deliberately crude, and everything it does is visible in the code:

    - Tissue is a flat sheet. A straight wavefront sweeps across it.
    - Each patch of tissue currently *in* the upstroke acts as a small current
      **dipole** pointing in the direction of travel — that is the standard
      approximation for a depolarisation front.
    - The potential at an electrode is the sum of those dipole contributions,
      falling off with distance.
    - Two electrodes, a couple of millimetres apart, sit just above the sheet.

    That is enough to reproduce the classic shapes: a **biphasic** unipolar
    deflection (rising as the wave approaches, crashing down as it passes under),
    and a bipolar signal that is the difference of the two.
    """)
    return


@app.cell
def _(np):
    # ILLUSTRATIVE. A dipole-sheet model of a propagating wavefront.
    def toy_electrograms(
        angle_deg=0.0,
        speed=0.6,
        height=1.5,
        spacing=2.0,
        far_field=0.0,
        half_width=20.0,
        step=0.5,
        upstroke=0.6,
        duration=60.0,
        dt=0.4,
    ):
        """Extracellular potential at two electrodes above a sheet of tissue.

        `angle_deg` is the angle between the direction the wave travels and the
        line joining the two electrodes. `far_field` adds a slow, broad signal
        from a distant source, which reaches both electrodes essentially equally.
        Returns (times, unipolar_a, unipolar_b, bipolar) in arbitrary units.
        """
        grid = np.arange(-half_width, half_width + step, step)
        gx, gy = np.meshgrid(grid, grid, indexing="ij")

        theta = np.radians(angle_deg)
        direction = np.array([np.cos(theta), np.sin(theta)])
        along = gx * direction[0] + gy * direction[1]

        times = np.arange(0.0, duration, dt)
        front = speed * times - half_width

        electrodes = np.array([[-spacing / 2, 0.0], [spacing / 2, 0.0]])
        traces = np.zeros((2, len(times)))

        for e, (ex, ey) in enumerate(electrodes):
            dx, dy = ex - gx, ey - gy
            r = np.sqrt(dx * dx + dy * dy + height * height)
            # dipole . r_hat / r^2, i.e. the potential of a dipole sheet
            geometry = (dx * direction[0] + dy * direction[1]) / r**3
            for t, position in enumerate(front):
                upstroke_profile = np.exp(-0.5 * ((along - position) / upstroke) ** 2)
                traces[e, t] = np.sum(upstroke_profile * geometry) * step * step

        traces = traces / (np.max(np.abs(traces)) or 1.0)

        if far_field:
            # A big, slow, distant source: nearly identical at both electrodes.
            centre, width = duration * 0.45, duration * 0.10
            shared = far_field * np.sin((times - centre) / width) * np.exp(
                -0.5 * ((times - centre) / width) ** 2
            )
            traces[0] += shared
            traces[1] += shared * 0.98

        return times, traces[0], traces[1], traces[0] - traces[1]

    return (toy_electrograms,)


@app.cell(hide_code=True)
def _(mo):
    toy_angle = mo.ui.slider(
        0, 90, step=5, value=0, label="angle between wave and electrode pair (°)",
        show_value=True, full_width=True,
    )
    toy_spacing = mo.ui.slider(
        0.5, 6.0, step=0.5, value=2.0, label="electrode spacing (mm)",
        show_value=True, full_width=True,
    )
    toy_far = mo.ui.slider(
        0.0, 2.0, step=0.25, value=0.0, label="far-field contamination",
        show_value=True, full_width=True,
    )
    mo.vstack(
        [
            mo.md("**Try it.** Move the sliders and watch the three traces change."),
            toy_angle,
            toy_spacing,
            toy_far,
        ]
    )
    return toy_angle, toy_far, toy_spacing


@app.cell(hide_code=True)
def _(plt, toy_angle, toy_electrograms, toy_far, toy_spacing):
    _t, _a, _b, _bip = toy_electrograms(
        angle_deg=toy_angle.value,
        spacing=toy_spacing.value,
        far_field=toy_far.value,
    )

    _fig, _axes = plt.subplots(3, 1, figsize=(10, 5.4), sharex=True)
    _axes[0].plot(_t, _a, color="#1f77b4", lw=1.4)
    _axes[0].set_ylabel("unipolar A")
    _axes[1].plot(_t, _b, color="#ff7f0e", lw=1.4)
    _axes[1].set_ylabel("unipolar B")
    _axes[2].plot(_t, _bip, color="#2ca02c", lw=1.6)
    _axes[2].set_ylabel("bipolar\n(A − B)")
    _axes[2].set_xlabel("time (ms)")
    _axes[0].set_title(
        f"Toy model — wave at {toy_angle.value}° to the electrode pair, "
        f"{toy_spacing.value:.1f} mm apart   |   "
        f"bipolar peak-to-peak {_bip.max() - _bip.min():.2f}, "
        f"unipolar {_a.max() - _a.min():.2f}"
    )
    for _ax in _axes:
        _ax.grid(alpha=0.25)
        _ax.axhline(0, color="grey", lw=0.6)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, takeaway, trap):
    mo.vstack(
        [
            mo.md(
                r"""
    Three things are worth doing with those sliders before moving on.

    1. **Leave everything at default.** The unipolar traces rise as the wave
       approaches, then crash steeply downward as it passes underneath. That steep
       downstroke is the moment of local activation, and it is what every annotation
       algorithm in Part 2 is trying to find.
    2. **Turn up the far-field.** Both unipolar traces get swamped by a slow bump
       that has nothing to do with the tissue under the electrode — but the bipolar
       trace barely notices, because subtracting two nearly-identical
       contaminations removes them. That is the entire reason bipolar recordings
       exist.
    3. **Take the angle to 90°.** The bipolar signal collapses to nothing, while
       both unipolar signals are unchanged. The wave arrives at both electrodes at
       the *same instant*, so the difference between them is zero.
    """
            ),
            trap(
                "That third one is not a modelling artefact — it is a real and "
                "well-known weakness of bipolar recordings, and it means perfectly "
                "healthy tissue can read as low voltage purely because of how the "
                "catheter happened to be lying. Part 6 sets up the test, and Part 7 "
                "measures how big the effect actually is in this dataset. The answer "
                "is smaller than the textbooks imply."
            ),
            takeaway(
                "A unipolar recording sees everything, local and distant. A bipolar "
                "recording sees mostly what is local, at the cost of caring about "
                "direction. Neither is 'the' signal; they are two different "
                "instruments."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now the real thing

    Same three traces, from the file. Pick any of the recorded points and look at
    what was actually captured — this is the raw material of everything that
    follows.
    """)
    return


@app.cell(hide_code=True)
def _(mo, n_points):
    point_choice = mo.ui.slider(
        0, n_points - 1, step=1, value=500, label="mapping point",
        show_value=True, full_width=True, debounce=True,
    )
    point_choice
    return (point_choice,)


@app.cell
def _(bipolar, bundle, fs, plt, point_choice, times_ms, unipolar):
    # MEASURED. One recorded point, three ways: two unipolar channels and the
    # bipolar difference the clinical system actually uses.
    _i = point_choice.value

    _fig, _axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    _axes[0].plot(times_ms, unipolar[_i, :, 0], lw=0.8, color="#1f77b4")
    _axes[0].set_ylabel("unipolar A\n(mV)")
    _axes[0].set_title(
        f"point {bundle.internal_names[_i]}  —  {fs:.0f} Hz, "
        f"{times_ms[-1] / 1000:.2f} s, bipolar range "
        f"{bipolar[_i].min():.2f} to {bipolar[_i].max():.2f} mV"
    )
    _axes[1].plot(times_ms, unipolar[_i, :, 1], lw=0.8, color="#ff7f0e")
    _axes[1].set_ylabel("unipolar B\n(mV)")
    _axes[2].plot(times_ms, bipolar[_i], lw=0.8, color="#2ca02c")
    _axes[2].set_ylabel("bipolar\n(mV)")
    _axes[2].set_xlabel("time (ms)")
    for _ax in _axes:
        _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(fs, mo, n_samples, plain):
    mo.vstack(
        [
            mo.md(
                rf"""
    Note the shape of the recording itself: **{fs:.0f} samples per second**, one
    every {1000 / fs:.1f} ms, for {n_samples / fs:.1f} seconds — so each "point" in
    this dataset is a {n_samples:,}-sample snippet containing several heartbeats,
    not a single number.

    The repeating structure is the patient's rhythm. Somewhere in each cycle is the
    deflection that belongs to *this* piece of tissue; the rest is the rest of the
    heart, arriving as far-field. Picking out the right deflection, in the right
    cycle, is Part 2.
    """
            ),
            plain(
                "Every point in the dataset is a few seconds of recording, not a "
                "single measurement. Turning that into one number — 'the wave "
                "arrived here at time T' — is a decision, and decisions can be made "
                "differently."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — The electrogram up close

    An **electrogram** is the name for one of those traces: a recording of the
    electrical field made from inside the heart. (An **ECG**, by contrast, is
    recorded from the skin. Same physics, thousands of times further away.)

    This part does two things. First it tests a textbook claim against the file and
    watches it fail. Then it turns a trace into the single number every map is built
    from — the moment of activation — and shows that this number is a *choice*.
    """)
    return


@app.cell(hide_code=True)
def _(deeper, mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Unipolar and bipolar, precisely

    - A **unipolar** recording is one electrode measured against a far-away
      reference — often a patch on the skin. It sees local activity plus everything
      else the heart is doing.
    - A **bipolar** recording is the *difference* between two electrodes a couple of
      millimetres apart on the same catheter. Distant sources reach both electrodes
      almost identically and cancel in the subtraction; local sources reach them
      differently and survive.

    The toy model in Part 1 showed both behaviours. The formal way to say it: taking
    the difference of two spatially offset samples of a field is a **discrete
    spatial derivative**, which is a high-pass filter in space. It buys locality,
    and charges for it in direction-sensitivity.
    """
            ),
            plain(
                "Two microphones side by side in a noisy room. Subtract one from the "
                "other and the room noise vanishes, because it hit both equally — "
                "what is left is whatever was happening right between them."
            ),
            deeper(
                "why clinicians still read the unipolar channel",
                r"""
    Because the two recordings fail in different ways, and a disagreement between
    them is informative.

    The unipolar signal has no electrode axis, so it has no direction blindness — it
    cannot be silenced by a wave arriving side-on. Its *shape* also carries
    information the bipolar cannot: a purely negative (QS) unipolar deflection means
    the wave started under the electrode and moved away in every direction, which is
    the signature of a focal source. An rS shape means the wave passed through.

    The price is that unipolar signals are dominated by far-field, drift with
    respiration, and depend on where the reference electrode is. In practice the
    bipolar channel drives the numbers and the unipolar channel is a sanity check.
    """,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Testing the textbook claim

    Every textbook says bipolar *is* electrode A minus electrode B. The file stores
    all three channels, so rather than believe it, we can check it. This takes two
    lines and is the most instructive failure in the notebook.
    """)
    return


@app.cell
def _(bipolar, n_points, np, unipolar):
    # DERIVED. Is the stored bipolar channel exactly uniA - uniB?
    derived_bipolar = unipolar[:, :, 0] - unipolar[:, :, 1]

    per_point_corr = np.array(
        [
            np.corrcoef(derived_bipolar[i], bipolar[i])[0, 1]
            if np.std(derived_bipolar[i]) > 0 and np.std(bipolar[i]) > 0
            else np.nan
            for i in range(n_points)
        ]
    )

    print(f"median per-point correlation : {np.nanmedian(per_point_corr):.3f}")
    print(
        f"points correlating > 0.99    : "
        f"{int(np.nansum(per_point_corr > 0.99))} of {n_points}"
    )
    print()
    print("amplitude check (99th percentile of |signal|):")
    print(f"  stored bipolar  : {np.nanpercentile(np.abs(bipolar), 99):.2f} mV")
    print(f"  uniA - uniB     : {np.nanpercentile(np.abs(derived_bipolar), 99):.2f} mV")
    print(
        f"  ratio           : "
        f"{np.nanpercentile(np.abs(derived_bipolar), 99) / np.nanpercentile(np.abs(bipolar), 99):.2f}"
    )
    print(
        f"  single unipolar : {np.nanpercentile(np.abs(unipolar[:, :, 0]), 99):.2f} mV"
        f"   <- far-field included"
    )
    return derived_bipolar, per_point_corr


@app.cell(hide_code=True)
def _(bipolar, derived_bipolar, mo, np, per_point_corr, plain, unipolar):
    mo.vstack(
        [
            mo.md(
                rf"""
    **The claim does not survive contact with the file.** The median correlation is
    about **{np.nanmedian(per_point_corr):.2f}**, and *not one* of the recorded
    points matches cleanly.

    But read the amplitude rows before concluding the theory is wrong. The
    subtraction lands within about
    {abs(1 - np.nanpercentile(np.abs(derived_bipolar), 99) / np.nanpercentile(np.abs(bipolar), 99)) * 100:.0f}%
    of the stored bipolar's scale, while a *single* unipolar channel is
    {np.nanpercentile(np.abs(unipolar[:, :, 0]), 99) / np.nanpercentile(np.abs(bipolar), 99):.1f}×
    larger. So the subtraction is doing exactly what the physics says — cancelling a
    large shared far-field component and leaving a small local signal. **The physics
    is right; the arithmetic identity is not.**

    The likely reason is mundane: the channels travel different processing paths
    before export. Bipolar channels are band-pass filtered (typically ~30–400 Hz) to
    sharpen local deflections, while unipolar channels keep far more low-frequency
    content. We can check how much of the gap that explains.
    """
            ),
            plain(
                "The bipolar trace in the file is not literally the difference of "
                "the two unipolar traces stored beside it — the manufacturer "
                "filtered them differently before saving. The idea is right; the "
                "equation is not."
            ),
        ]
    )
    return


@app.cell
def _(bipolar, derived_bipolar, fs, np, sps):
    # DERIVED. Does matching the filtering close the gap?
    def median_correlation(candidate, n=400):
        scores = [
            np.corrcoef(candidate[i], bipolar[i])[0, 1]
            for i in range(n)
            if np.std(candidate[i]) > 0
        ]
        return np.nanmedian(scores)

    print(f"{'signal compared against stored bipolar':<42}{'median correlation':>20}")
    print(f"{'uniA - uniB, unfiltered':<42}{median_correlation(derived_bipolar):>20.3f}")
    for _low, _high in [(10, 400), (30, 300), (30, 400)]:
        _b, _a = sps.butter(
            2, [_low / (fs / 2), _high / (fs / 2)], btype="bandpass"
        )
        _filtered = sps.filtfilt(_b, _a, derived_bipolar[:400], axis=1)
        print(
            f"{f'uniA - uniB, band-pass {_low}-{_high} Hz':<42}"
            f"{median_correlation(_filtered):>20.3f}"
        )
    return


@app.cell(hide_code=True)
def _(bipolar, deeper, derived_bipolar, mo, np, sps, takeaway):
    def _best_lag_correlations(n=200, max_lag=50):
        """Best correlation over any time shift up to +/- max_lag samples."""
        best = []
        for i in range(n):
            a = derived_bipolar[i] - derived_bipolar[i].mean()
            c = bipolar[i] - bipolar[i].mean()
            denominator = np.sqrt((a**2).sum() * (c**2).sum())
            if denominator <= 0:
                continue
            full = sps.correlate(c, a, mode="full") / denominator
            middle = len(a) - 1
            best.append(np.max(full[middle - max_lag : middle + max_lag + 1]))
        return np.array(best)

    _best = _best_lag_correlations()

    mo.vstack(
        [
            mo.md(
                r"""
    Filtering moves the correlation from roughly 0.51 to 0.61. Real, but nowhere
    near closing the gap.
    """
            ),
            deeper(
                "is it just a clock offset between the channels?",
                rf"""
    No. Allowing each point its own optimal time shift — up to ±50 ms, which is far
    more than any plausible export latency — lifts the median correlation only to
    **{np.median(_best):.2f}**, and just **{np.mean(_best > 0.9) * 100:.0f}%** of
    points exceed 0.9. A simple misalignment would have snapped almost all of them
    into place. Whatever the systems did to these channels, it was not a delay.
    """,
            ),
            takeaway(
                "A clinical export is not a set of arithmetically consistent "
                "arrays. Each channel has been through vendor filtering, gain and "
                "annotation logic that the file does not document. Verify "
                "relationships in the data rather than assuming them from theory — "
                "and when they fail, check whether the *concept* still holds before "
                "throwing it out. Every later part of this notebook computes from "
                "the stored bipolar channel, because that is the one the clinical "
                "measurements were actually derived from."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(jargon, mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### From a trace to a number: what time did the wave arrive?

    A map needs one number per point, not a 2.5-second recording. That number is
    the **local activation time (LAT)**: when the wave reached this piece of tissue.

    "When" needs a zero. A trace on its own tells you a wave arrived, not *where in
    the heartbeat* it arrived, and the recordings were made minutes apart at
    different points — so timing is always measured relative to a **reference
    channel**, usually a stable electrode left parked somewhere in the heart for the
    whole procedure. Same beat, same zero, every point.

    Three numbers per point make this work:

    | field | meaning |
    |:---|:---|
    | `referenceAnnot` | the fiducial moment on the reference channel — the metronome tick |
    | `mapAnnot` | the moment of local activation at this point, chosen by the system |
    | `woi` | the **window of interest**: the slice of the recording allowed to contain a valid activation |

    **Local activation time is `mapAnnot − referenceAnnot`.** Negative means this
    tissue activated *before* the reference; positive, after.
    """
            ),
            plain(
                "Every point is timed against a metronome tick so that recordings "
                "made minutes apart can be compared on one map. The window of "
                "interest is the operator saying 'only look for the beat in here'."
            ),
            jargon(
                [
                    ("electrogram", "a voltage trace recorded from inside the heart"),
                    ("ECG", "a voltage trace recorded from the skin"),
                    ("LAT", "local activation time — when the wave reached one point"),
                    ("annotation", "the chosen instant on a trace that counts as 'activation'"),
                    ("fiducial", "a fixed reference instant, the zero of the time axis"),
                    ("window of interest (WOI)", "the time range allowed to contain a valid activation"),
                    ("far-field", "signal from tissue far from the electrode"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, n_points):
    annot_point = mo.ui.slider(
        0, n_points - 1, step=1, value=500, label="mapping point",
        show_value=True, full_width=True, debounce=True,
    )
    mo.vstack(
        [
            mo.md(
                "**The three numbers, drawn on the trace.** Shaded band: the window "
                "of interest. Dashed line: the reference tick. Solid line: the "
                "system's activation annotation."
            ),
            annot_point,
        ]
    )
    return (annot_point,)


@app.cell
def _(annot_point, bipolar, bundle, plt, times_ms):
    # MEASURED. Where the system placed the annotation, and inside what window.
    _i = annot_point.value
    _ref = float(bundle.reference_activation_time[_i])
    _map = float(bundle.local_activation_time[_i])
    _woi = bundle.window_of_interest[_i]

    _fig, _ax = plt.subplots(figsize=(11, 3.4))
    _ax.plot(times_ms, bipolar[_i], lw=0.8, color="#2ca02c")
    _ax.axvspan(_ref + _woi[0], _ref + _woi[1], color="#4c72b0", alpha=0.15,
                label="window of interest")
    _ax.axvline(_ref, color="grey", ls="--", lw=1.2, label="reference tick")
    if _map > -1000:
        _ax.axvline(_map, color="crimson", lw=1.4, label="system annotation")
    _ax.set_xlim(_ref - 400, _ref + 400)
    _ax.set_xlabel("time (ms)")
    _ax.set_ylabel("bipolar (mV)")
    _ax.set_title(
        f"point {bundle.internal_names[_i]}  —  LAT = "
        f"{_map - _ref:.0f} ms relative to the reference"
        if _map > -1000
        else f"point {bundle.internal_names[_i]}  —  never annotated"
    )
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### And here is a trap that will silently ruin a map

    Some points were never annotated — the system could not find a defensible
    activation, or the operator rejected it. Those points do not carry a NaN or a
    missing-value flag. They carry **−10000**.
    """)
    return


@app.cell
def _(bundle, epkit, np):
    # MEASURED, then cleaned. Unannotated points carry a -10000 sentinel.
    raw_lat = bundle.local_activation_time - bundle.reference_activation_time
    lat, valid = epkit.clean_lat(
        bundle.local_activation_time, bundle.reference_activation_time
    )

    print(f"raw LAT range     : {raw_lat.min():.0f} to {raw_lat.max():.0f} ms")
    print(f"sentinel points   : {int(np.sum(raw_lat <= -1000))} of {raw_lat.size}")
    print(f"cleaned LAT range : {np.nanmin(lat):.0f} to {np.nanmax(lat):.0f} ms")
    print()
    print(f"mean WITH sentinels    : {raw_lat.mean():>10.1f} ms   <- meaningless")
    print(f"mean WITHOUT sentinels : {np.nanmean(lat):>10.1f} ms   <- correct")
    return lat, raw_lat, valid


@app.cell(hide_code=True)
def _(mo, np, raw_lat, trap):
    mo.vstack(
        [
            mo.md(
                rf"""
    **{int(np.sum(raw_lat <= -1000))} points out of {raw_lat.size:,}** —
    {np.mean(raw_lat <= -1000) * 100:.1f}% of the data — drag the naive mean from a
    perfectly reasonable {np.nanmean(np.where(raw_lat > -1000, raw_lat, np.nan)):.0f} ms
    to a physically impossible {raw_lat.mean():.0f} ms.

    A heartbeat is under a second. A local activation time of minus ninety
    milliseconds is fine; minus eighty-nine *thousand* would mean the tissue
    activated a minute and a half before the reference beat.
    """
            ),
            trap(
                "Any statistic, interpolation, or colour scale computed without "
                "masking these is wrong — and wrong in a way that still produces a "
                "plausible-looking picture. Sentinel values are the single most "
                "common way to silently corrupt an activation map, and they are not "
                "unique to electrophysiology: −999, −9999 and 0 do the same damage "
                "in every scientific file format ever written."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Re-annotating it ourselves

    The system chose `mapAnnot` for us using an algorithm it does not disclose. The
    conventional rule for a bipolar electrogram is to annotate at the **steepest
    deflection** (or the peak absolute amplitude) within the window of interest.
    Let's apply that rule ourselves and see how far we land from the system.
    """)
    return


@app.cell
def _(bipolar, bundle, lat, n_points, n_samples, np, valid):
    # DERIVED. Re-annotate by steepest deflection within the window of interest.
    woi = bundle.window_of_interest
    ref_annot = bundle.reference_activation_time

    our_lat = np.full(n_points, np.nan)
    for _i in range(n_points):
        _start = int(ref_annot[_i] + woi[_i, 0])
        _stop = int(ref_annot[_i] + woi[_i, 1])
        _start, _stop = max(_start, 0), min(_stop, n_samples)
        if _stop - _start < 3:
            continue
        _segment = bipolar[_i, _start:_stop]
        our_lat[_i] = (
            _start + int(np.argmax(np.abs(np.gradient(_segment)))) - ref_annot[_i]
        )

    annotation_pairs = valid & np.isfinite(our_lat)
    annotation_gap = our_lat[annotation_pairs] - lat[annotation_pairs]

    print(f"points compared   : {annotation_pairs.sum()}")
    print(f"median difference : {np.median(annotation_gap):>7.1f} ms")
    print(
        f"IQR               : {np.percentile(annotation_gap, 25):.1f} to "
        f"{np.percentile(annotation_gap, 75):.1f} ms"
    )
    print(f"within +/-5 ms    : {np.mean(np.abs(annotation_gap) <= 5) * 100:.1f}%")
    print(
        f"worst 1%          : beyond "
        f"{np.percentile(np.abs(annotation_gap), 99):.0f} ms, "
        f"max {np.max(np.abs(annotation_gap)):.0f} ms"
    )
    return annotation_gap, annotation_pairs, our_lat, ref_annot, woi


@app.cell(hide_code=True)
def _(annotation_gap, mo, np, takeaway):
    mo.vstack(
        [
            mo.md(
                rf"""
    A crude one-line rule reproduces an undocumented commercial algorithm to the
    millisecond for **{np.mean(np.abs(annotation_gap) <= 5) * 100:.0f}%** of points.
    That is genuinely reassuring: the convention is doing the work, not a secret.

    The interesting part is the other
    {100 - np.mean(np.abs(annotation_gap) <= 5) * 100:.0f}%, where the two rules
    disagree by up to {np.max(np.abs(annotation_gap)):.0f} ms — comparable to the
    entire spread of activation across the chamber. Those disagreements are not
    scattered at random. Part 8 comes back with a way to predict exactly which
    points they will be.
    """
            ),
            takeaway(
                "Where an electrogram is a single sharp spike, every annotation rule "
                "agrees. Where it is fragmented — which is precisely the diseased "
                "tissue we most want to characterise — the rules diverge, and that "
                "uncertainty propagates into every map downstream."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — The shell: where the 3D heart comes from

    The coloured heart on an EP screen looks like a scan. It is not. Nobody put this
    patient in an MRI to make it; it was built during the procedure, from the
    catheter itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo, plain, trap):
    mo.vstack(
        [
            mo.md(
                r"""
    The operator sweeps the catheter around the inside of the chamber. The tracking
    system knows where the tip is at all times, and it accumulates the *outermost*
    positions the tip reached, wrapping a surface around them. This is **fast
    anatomical mapping (FAM)**, and the result is a **triangulated surface mesh**:
    a list of 3D vertices, plus a list of triangles joining them.

    So the shell is, literally, the envelope of where the catheter has been.
    """
            ),
            plain(
                "Imagine mapping a cave in total darkness by walking around and "
                "touching the walls, then stretching a sheet over everywhere your "
                "hand reached. That is the 3D heart on the screen."
            ),
            trap(
                "Two consequences follow, and both bite later. **Anatomy the "
                "catheter never touched does not exist** in the model — not shown as "
                "uncertain, simply absent. And the geometry and the electrical "
                "recordings are two separate acquisitions at completely different "
                "densities: thousands of vertices, hundreds of recordings."
            ),
        ]
    )
    return


@app.cell
def _(egm_points, np, points, spatial, triangles):
    # MEASURED / DERIVED. Mesh resolution against electrical sampling density.
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)

    triangle_points = points[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        ),
        axis=1,
    )

    nn_distance = spatial.cKDTree(egm_points).query(egm_points, k=2)[0][:, 1]

    print(f"Mesh vertices             : {len(points):,}")
    print(f"Mesh triangles            : {len(triangles):,}")
    print(f"Total surface area        : {areas.sum():,.0f} mm^2")
    print(f"Median mesh edge length   : {np.median(edge_lengths):.2f} mm   <- FAM resolution")
    print()
    print(f"Electrical points         : {len(egm_points):,}")
    print(f"Median point-to-point     : {np.median(nn_distance):.2f} mm   <- nearest neighbour")
    print()
    print(f"Vertices per point        : {len(points) / len(egm_points):.1f}")
    print(f"Area per electrical point : {areas.sum() / len(egm_points):.0f} mm^2")
    return areas, edge_lengths, nn_distance


@app.cell(hide_code=True)
def _(areas, edge_lengths, mo, nn_distance, np, plain):
    mo.vstack(
        [
            mo.md(
                rf"""
    For scale: **{areas.sum() / 100:.0f} cm² of inner surface**, about the area of
    two and a half credit cards, described by a mesh whose triangles are about
    {np.median(edge_lengths):.1f} mm on a side.

    Now the first surprise. Median mesh edge is
    {np.median(edge_lengths):.2f} mm and the median distance from one electrical
    point to its nearest neighbour is {np.median(nn_distance):.2f} mm — practically
    identical. It is tempting to conclude the two acquisitions are equally well
    resolved.

    **That conclusion would be wrong**, and the reason is worth dwelling on.

    Nearest-neighbour distance measures how close points are *to each other*, not
    how well they cover the surface. A multi-electrode catheter drops ten points at
    once in a tight cluster, so every point has a close neighbour — even if the
    whole cluster sits in a region already sampled to death while half the chamber
    has never been touched. Part 9 shows the catheter that did exactly this.

    The metric that actually matters is the reverse question: **for each vertex, how
    far away is the nearest real measurement?** That is what decides whether a
    vertex can be honestly coloured.
    """
            ),
            plain(
                "Ten thermometers in one garden do not tell you the temperature of "
                "the county. What matters is not how close the thermometers are to "
                "each other, but how far the nearest thermometer is from the place "
                "you want to know about."
            ),
        ]
    )
    return


@app.cell
def _(np, plt, points, spatial, surface_points):
    # DERIVED. Coverage, measured from the surface's point of view.
    coverage_distance = spatial.cKDTree(surface_points).query(points, k=1)[0]

    print("distance from each vertex to the nearest electrical measurement:")
    for _pct in (10, 25, 50, 75, 90, 99):
        print(f"  {_pct:>2}th percentile : {np.percentile(coverage_distance, _pct):6.2f} mm")
    print()
    for _threshold in (3, 5, 10, 20):
        print(
            f"  vertices within {_threshold:>2} mm of a measurement : "
            f"{np.mean(coverage_distance <= _threshold) * 100:5.1f}%"
        )

    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.hist(coverage_distance, bins=60, color="#937860", edgecolor="white", lw=0.4)
    _ax.axvline(10, color="crimson", ls="--", label="10 mm — a typical interpolation cutoff")
    _ax.set_xlabel("distance from vertex to nearest measurement (mm)")
    _ax.set_ylabel("vertices")
    _ax.set_title("Surface coverage: how far the map has to reach")
    _ax.legend()
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return (coverage_distance,)


@app.cell(hide_code=True)
def _(coverage_distance, mo, np, takeaway):
    mo.vstack(
        [
            mo.md(
                rf"""
    That is the real picture, and it is far less flattering than 1.5 mm suggested.
    Coverage is deeply uneven: {np.mean(coverage_distance <= 3) * 100:.0f}% of the
    surface sits within 3 mm of a measurement, while
    {np.mean(coverage_distance > 10) * 100:.0f}% is more than a centimetre from any.
    Those distant vertices are the ones that will go grey in Part 4 — and the ones
    an over-eager interpolator would happily colour in anyway.

    Below, the same number painted onto the chamber. **Drag to rotate, scroll to
    zoom.** Dark regions are where the map is entitled to an opinion; bright regions
    are where it is guessing.
    """
            ),
            takeaway(
                "Nearest-neighbour spacing flatters clustered sampling. It is the "
                "number most easily quoted about a map and among the least "
                "informative. Ask instead: how far is the nearest real measurement "
                "from the place I am reading?"
            ),
        ]
    )
    return


@app.cell
def _(coverage_distance, epkit, np, points, triangles):
    # DERIVED. Coverage as a map in its own right.
    epkit.plotly_figure(
        [
            epkit.mesh_to_plotly(
                points,
                triangles,
                scalars=np.clip(coverage_distance, 0, 20),
                colorscale="Cividis",
                cmin=0,
                cmax=20,
                colorbar_title="mm to nearest<br>measurement",
                name="coverage",
            )
        ],
        title="Where the measurements actually are (dark = well sampled)",
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Where the catheter sat: projection distance

    An electrical point is recorded wherever the catheter tip happened to be, which
    is near the wall but never exactly on it — the tip floats in blood, the wall
    moves with every heartbeat and every breath. The system therefore projects each
    recording onto the nearest point of the shell before using it.

    That projection distance is a quality signal in its own right: a point recorded
    far from the wall was probably not in good contact, and a recording made without
    contact is mostly far-field.
    """
            ),
            plain(
                "Each recording has to be assigned a home on the 3D surface. How far "
                "it had to be moved to get there tells you how much to trust it."
            ),
        ]
    )
    return


@app.cell
def _(egm_points, np, plt, surface_points):
    # MEASURED. Distance from the recording position to its projection on the shell.
    projection_distance = np.linalg.norm(egm_points - surface_points, axis=1)

    print(f"median : {np.nanmedian(projection_distance):.2f} mm")
    print(f"p95    : {np.nanpercentile(projection_distance, 95):.2f} mm")
    print(f"max    : {np.nanmax(projection_distance):.2f} mm")

    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.hist(projection_distance, bins=50, color="#4c72b0", edgecolor="white", lw=0.4)
    _ax.axvline(
        np.nanmedian(projection_distance), color="crimson", ls="--",
        label=f"median {np.nanmedian(projection_distance):.2f} mm",
    )
    _ax.set_xlabel("distance from recording site to shell (mm)")
    _ax.set_ylabel("points")
    _ax.set_title("How far each electrical point sat from the wall")
    _ax.legend()
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### The holes are anatomy

    The shell is not a closed bag. Where the chamber opens into a vein or a valve,
    the mesh simply ends, leaving a **free boundary** — an edge that belongs to only
    one triangle instead of the usual two.

    Finding them needs no anatomy and no clinical knowledge at all. Count how many
    triangles use each edge; the edges used once are the rim of a hole. It is pure
    topology, and it recovers real anatomical structures.

    From the schematic in Part 0, a left atrium should have about six openings: the
    mitral valve (much the largest), four pulmonary veins, and the appendage.
    """
            ),
            plain(
                "A hole in the mesh is not missing data. It is a doorway — a vein or "
                "a valve — and we can find every one of them by asking which edges "
                "have a triangle on only one side."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np, triangles):
    # DERIVED, cross-checked against OpenEP's own routine.
    from collections import Counter

    edge_use = Counter()
    for _tri in triangles:
        for _a, _b in ((_tri[0], _tri[1]), (_tri[1], _tri[2]), (_tri[2], _tri[0])):
            edge_use[(min(_a, _b), max(_a, _b))] += 1

    our_boundary_edges = [edge for edge, count in edge_use.items() if count == 1]
    boundary_rings = bundle.boundary_rings
    openep_edges = sum(len(ring) - 1 for ring in boundary_rings)

    print(f"our boundary edges    : {len(our_boundary_edges)}")
    print(f"OpenEP boundary edges : {openep_edges}")
    print(f"distinct boundaries   : {len(boundary_rings)}")
    print()

    boundary_order = np.argsort(bundle.boundary_lengths)[::-1]
    print(f"{'#':<3}{'edges':>7}{'perimeter mm':>15}{'area mm^2':>12}   likely structure")
    for _rank, _idx in enumerate(boundary_order):
        _guess = "mitral valve" if _rank == 0 else "pulmonary vein / appendage"
        print(
            f"{_rank:<3}{len(boundary_rings[_idx]) - 1:>7}"
            f"{bundle.boundary_lengths[_idx]:>15.1f}"
            f"{bundle.boundary_areas[_idx]:>12.1f}   {_guess}"
        )
    return boundary_order, boundary_rings


@app.cell(hide_code=True)
def _(boundary_rings, bundle, mo, np, takeaway):
    mo.vstack(
        [
            mo.md(
                rf"""
    Our count matches OpenEP's exactly, and the structure is what the anatomy
    predicts: **{len(boundary_rings)} openings**, one of them
    ({np.max(bundle.boundary_lengths):.0f} mm around, enclosing
    {np.max(bundle.boundary_areas):.0f} mm²) more than twice the perimeter of any
    other — the mitral valve — and five smaller ones for the four pulmonary veins
    and the appendage.
    """
            ),
            takeaway(
                "We identified cardiac anatomy from nothing but mesh connectivity. "
                "No clinical input, no labels in the file, no machine learning — "
                "just counting how many triangles use each edge."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The first render

    Bare anatomy, with the openings traced on top. No physiology yet — this is the
    canvas everything else gets painted onto.

    **Drag to rotate.** The four vein openings and the big valve opening should be
    recognisable from the schematic in Part 0.
    """)
    return


@app.cell
def _(boundary_order, boundary_rings, bundle, epkit, go, points, triangles):
    # DERIVED. The shell, with each free boundary traced.
    _shell = epkit.mesh_to_plotly(points, triangles, name="atrial shell")

    _palette = ["#e45756", "#4c78a8", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]
    _traces = [_shell]
    for _rank, _idx in enumerate(boundary_order):
        _ring = points[boundary_rings[_idx]]
        _traces.append(
            go.Scatter3d(
                x=_ring[:, 0], y=_ring[:, 1], z=_ring[:, 2],
                mode="lines",
                line=dict(width=6, color=_palette[_rank % len(_palette)]),
                name=f"opening {_rank} ({bundle.boundary_lengths[_idx]:.0f} mm)",
            )
        )

    epkit.plotly_figure(
        _traces, title="Atrial shell with its free boundaries (veins and valve)"
    )
    return


@app.cell(hide_code=True)
def _(mo, n_points, plain, points):
    mo.vstack(
        [
            mo.md(
                rf"""
    ## Part 4 — From points to maps: the interpolation problem

    This is the heart of the matter, and if you read only one part, read this one.

    We have **{n_points:,} measurements** and **{len(points):,} vertices** to
    colour. Every vertex that was not measured gets a value **invented by an
    interpolator** — a rule for guessing between the measurements.

    The standard rule is *inverse-distance weighting*: a vertex takes a weighted
    average of the nearby measurements, with closer ones counting for more.
    Crucially, it is also **distance-masked** — beyond some radius from any
    measurement, the interpolator refuses to guess and returns NaN, leaving a hole
    in the map.

    That refusal is the most honest thing the algorithm does, and the radius at
    which it refuses is a number *somebody chose*.
    """
            ),
            plain(
                "This is weather mapping. A handful of weather stations report real "
                "temperatures; the smooth coloured map on the forecast is drawn by "
                "guessing everywhere in between. The guess is usually reasonable. "
                "It is never a measurement — and how far the mapmaker is willing to "
                "guess from the nearest station is a setting, not a fact."
            ),
        ]
    )
    return


@app.cell
def _(np, spatial):
    # DERIVED. Inverse-distance interpolation with an explicit distance mask.
    def interpolate_to_surface(
        sample_points, sample_values, target_points, max_distance=10.0, k=8, power=2.0
    ):
        """Inverse-distance weighted interpolation with a hard distance cutoff.

        Vertices with no sample within `max_distance` return NaN rather than a
        fabricated value. That is the whole design: refusing to answer is a
        legitimate answer.
        """
        finite = np.isfinite(sample_values)
        sample_points = sample_points[finite]
        sample_values = sample_values[finite]

        tree = spatial.cKDTree(sample_points)
        k = min(k, len(sample_points))
        distances, indices = tree.query(target_points, k=k)
        distances = np.atleast_2d(distances.T).T
        indices = np.atleast_2d(indices.T).T

        with np.errstate(divide="ignore"):
            weights = 1.0 / np.power(distances, power)
        weights[~np.isfinite(weights)] = 1e12  # an exact hit dominates

        values = np.sum(weights * sample_values[indices], axis=1) / np.sum(
            weights, axis=1
        )
        values[distances[:, 0] > max_distance] = np.nan
        return values

    return (interpolate_to_surface,)


@app.cell
def _(bundle, interpolate_to_surface, lat, np, points, surface_points, valid):
    # DERIVED. The activation field, at a 10 mm cutoff.
    vertex_lat = interpolate_to_surface(surface_points[valid], lat[valid], points)
    openep_nan_share = np.isnan(bundle.openep_lat_field).mean() * 100

    print(f"vertices coloured : {np.isfinite(vertex_lat).sum():,} of {len(points):,}")
    print(f"our map, blank    : {np.isnan(vertex_lat).mean() * 100:.1f}%   (10 mm cutoff)")
    print(f"OpenEP's own map  : {openep_nan_share:.1f}% blank")
    return openep_nan_share, vertex_lat


@app.cell(hide_code=True)
def _(mo, np, openep_nan_share, vertex_lat):
    mo.md(
        rf"""
    The two disagree — {np.isnan(vertex_lat).mean() * 100:.1f}% against
    {openep_nan_share:.1f}% — and the disagreement is *entirely* our choice of
    cutoff. OpenEP, the reference implementation, is stricter about how far it will
    reach from a measurement than our 10 mm default.

    We never see OpenEP's threshold; it is buried in its code. But we can recover it
    by sweeping ours until the blank fraction matches.
    """
    )
    return


@app.cell
def _(np, openep_nan_share, points, spatial, surface_points, valid):
    # DERIVED. Which cutoff reproduces OpenEP's coverage?
    vertex_gap = spatial.cKDTree(surface_points[valid]).query(points, k=1)[0]

    print(f"{'cutoff':>8}{'blank %':>10}")
    for _threshold in (3, 4, 4.5, 5, 6, 8, 10):
        print(f"{_threshold:>5} mm{np.mean(vertex_gap > _threshold) * 100:>10.1f}")
    print(f"\nOpenEP's actual   {openep_nan_share:>8.1f}")
    return


@app.cell(hide_code=True)
def _(mo, np, openep_nan_share, takeaway, trap, vertex_lat):
    mo.vstack(
        [
            mo.md(
                rf"""
    So OpenEP will not colour a vertex more than roughly **4.5 mm** from a real
    measurement, and on that basis **{openep_nan_share:.0f}% of this chamber cannot
    be honestly coloured at all** — two of every five vertices, blank.

    That number is not a property of the heart, the catheter, or the recording. It
    is a threshold someone chose. Move it and the map's apparent completeness moves
    with it, while the underlying measurements stay exactly the same.
    """
            ),
            trap(
                "A map with no blank regions is not a better map. It is either a "
                "better-sampled chamber or a looser threshold, and from the picture "
                "alone you cannot tell which."
            ),
            takeaway(
                "Two implementations of 'the same' activation map, on identical "
                f"measurements, disagreed about "
                f"{openep_nan_share - np.isnan(vertex_lat).mean() * 100:.0f}% of the "
                "surface — purely over how far a guess may reach."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See it for yourself

    Move the cutoff and watch the map fill in. Nothing about the patient changes;
    nothing about the measurements changes. Only the algorithm's willingness to
    extrapolate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cutoff_mm = mo.ui.slider(
        2.0, 30.0, step=0.5, value=10.0,
        label="interpolation cutoff (mm)", show_value=True, full_width=True,
        debounce=True,
    )
    cutoff_mm
    return (cutoff_mm,)


@app.cell
def _(
    cutoff_mm,
    epkit,
    interpolate_to_surface,
    lat,
    mo,
    np,
    points,
    surface_points,
    triangles,
    valid,
):
    # DERIVED. The same measurements, at whatever cutoff you asked for.
    _field = interpolate_to_surface(
        surface_points[valid], lat[valid], points, max_distance=cutoff_mm.value
    )
    _coloured = np.isfinite(_field).mean() * 100

    mo.vstack(
        [
            mo.md(
                f"**{_coloured:.1f}% of the surface coloured** at a "
                f"{cutoff_mm.value:.1f} mm cutoff "
                f"({np.isfinite(_field).sum():,} of {len(points):,} vertices). "
                f"OpenEP's own choice is near 4.5 mm; a typical clinical system "
                f"draws considerably more than that."
            ),
            epkit.plotly_figure(
                [
                    epkit.mesh_to_plotly(
                        points, triangles, scalars=_field, colorscale="Turbo_r",
                        cmin=float(np.nanmin(lat)), cmax=float(np.nanmax(lat)),
                        colorbar_title="LAT (ms)",
                    )
                ],
                title=f"Activation map at a {cutoff_mm.value:.1f} mm cutoff",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Three reasonable choices, side by side

    Strict inverse-distance (10 mm), loose inverse-distance (30 mm), and
    nearest-neighbour — which never refuses, and colours every vertex with the value
    of whatever measurement happens to be closest.
    """)
    return


@app.cell
def _(
    interpolate_to_surface,
    lat,
    np,
    points,
    surface_points,
    valid,
    vertex_lat,
):
    # DERIVED. The same data under three interpolation regimes.
    from scipy.interpolate import griddata

    variants = {
        "IDW, 10 mm mask": vertex_lat,
        "IDW, 30 mm mask": interpolate_to_surface(
            surface_points[valid], lat[valid], points, max_distance=30.0
        ),
        "Nearest neighbour": griddata(
            surface_points[valid], lat[valid], points, method="nearest"
        ),
    }

    for _name, _field in variants.items():
        print(
            f"{_name:<20} coloured {np.isfinite(_field).mean() * 100:5.1f}%   "
            f"range {np.nanmin(_field):6.1f} to {np.nanmax(_field):6.1f} ms"
        )

    _strict, _loose = variants["IDW, 10 mm mask"], variants["IDW, 30 mm mask"]
    _nearest = variants["Nearest neighbour"]

    both_defined = np.isfinite(_strict) & np.isfinite(_loose)
    invented = np.isnan(_strict) & np.isfinite(_loose)
    nn_defined = np.isfinite(_strict) & np.isfinite(_nearest)

    print(
        f"\nwhere both are defined : median difference "
        f"{np.nanmedian(np.abs(_strict[both_defined] - _loose[both_defined])):.2f} ms"
    )
    print(
        f"vertices the 30 mm mask invents that 10 mm refused : "
        f"{invented.sum():,} ({invented.mean() * 100:.1f}% of the surface)"
    )
    print(
        f"nearest-neighbour vs IDW, where both defined : median "
        f"{np.nanmedian(np.abs(_strict[nn_defined] - _nearest[nn_defined])):.2f} ms"
    )
    return invented, variants


@app.cell
def _(epkit, lat, np, points, triangles, variants):
    # Three maps, shifted apart so they can be compared in one scene.
    _traces = []
    for _offset, (_name, _field) in enumerate(variants.items()):
        _shifted = points.copy()
        _shifted[:, 0] += _offset * 145.0
        _traces.append(
            epkit.mesh_to_plotly(
                _shifted, triangles, scalars=_field, colorscale="Turbo_r",
                cmin=float(np.nanmin(lat)), cmax=float(np.nanmax(lat)),
                name=_name, showscale=(_offset == 0), colorbar_title="LAT (ms)",
            )
        )

    epkit.plotly_figure(
        _traces,
        title="Same measurements, three interpolators — "
        "left: 10 mm | centre: 30 mm | right: nearest neighbour",
    )
    return


@app.cell(hide_code=True)
def _(invented, mo, takeaway):
    mo.vstack(
        [
            mo.md(
                rf"""
    Read that first number carefully: **where both interpolators are willing to
    commit, they agree exactly** — a median difference of zero milliseconds.
    Loosening the mask does not alter a single existing value.

    What it changes is the **{invented.mean() * 100:.0f}% of the surface**
    ({invented.sum():,} vertices) that the strict version left blank and the loose
    version fills in, conjured from measurements up to three centimetres away.

    That is the whole point. The left-hand map admits what it does not know. The
    middle one is smoother, prettier, and partly fiction. The right-hand one is
    confident everywhere and honest nowhere.
    """
            ),
            takeaway(
                "None of the three is more 'correct' than the others. They encode "
                "different assumptions about how far a measurement's influence "
                "should reach. A clinician reading a map is reading those "
                "assumptions as much as the tissue."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watching the map dissolve

    A different question: not *how far may we guess*, but *how much did we measure*.
    Throw measurements away at random and rebuild the map from what is left.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    keep_points = mo.ui.slider(
        steps=[25, 50, 100, 150, 250, 400, 700, 1344],
        value=150,
        label="measurements kept",
        show_value=True,
        full_width=True,
        debounce=True,
    )
    keep_points
    return (keep_points,)


@app.cell
def _(
    epkit,
    interpolate_to_surface,
    keep_points,
    lat,
    mo,
    np,
    points,
    surface_points,
    triangles,
    valid,
    vertex_lat,
):
    # DERIVED. Degrade the sampling density and watch the map degrade with it.
    _rng = np.random.default_rng(42)
    _valid_indices = np.flatnonzero(valid)
    _keep = _rng.choice(
        _valid_indices, size=min(keep_points.value, len(_valid_indices)), replace=False
    )
    _sparse = interpolate_to_surface(surface_points[_keep], lat[_keep], points)

    _traces = []
    for _offset, (_name, _field) in enumerate(
        [(f"{len(_keep)} points", _sparse), ("all measurements", vertex_lat)]
    ):
        _shifted = points.copy()
        _shifted[:, 0] += _offset * 145.0
        _traces.append(
            epkit.mesh_to_plotly(
                _shifted, triangles, scalars=_field, colorscale="Turbo_r",
                cmin=float(np.nanmin(lat)), cmax=float(np.nanmax(lat)),
                name=_name, showscale=(_offset == 0), colorbar_title="LAT (ms)",
            )
        )

    mo.vstack(
        [
            mo.md(
                f"**{len(_keep)} of {len(_valid_indices):,} measurements** — "
                f"{np.isfinite(_sparse).mean() * 100:.1f}% of the surface still "
                f"coloured, against "
                f"{np.isfinite(vertex_lat).mean() * 100:.1f}% with everything."
            ),
            epkit.plotly_figure(
                _traces, title="Left: degraded sampling.  Right: the full map."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, takeaway):
    mo.vstack(
        [
            mo.md(
                r"""
    Drag it down and watch what survives. The **overall pattern** — where activation
    starts, roughly where it ends — survives surprisingly far down; at 150 points,
    a ninth of the data, the big picture is still recognisable.

    **Fine structure does not.** The narrow corridor of slow conduction, the small
    early site, the gap in an ablation line: those are the first things to go, and
    they are exactly what an arrhythmia circuit is made of.
    """
            ),
            takeaway(
                "A sparsely sampled map can look perfectly convincing while having "
                "lost the thing you are looking for. Convincing and complete are "
                "different properties, and only one of them is visible."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Part 5 — The activation map

    Now the payoff. Colour the shell by **when** each region activated, and the
    propagation of a heartbeat across the chamber becomes visible.

    The clinical convention is **red early, blue late**, running through the
    spectrum in between. (This notebook uses `Turbo_r`, which is that convention.)
    """
            ),
            plain(
                "Red is where the wave arrived first, blue where it arrived last, "
                "and the colours in between are the journey. You are looking at a "
                "single heartbeat, frozen."
            ),
        ]
    )
    return


@app.cell
def _(lat, np, vertex_lat):
    # DERIVED. Activation timing across the chamber.
    earliest_vertex = int(np.nanargmin(vertex_lat))
    latest_vertex = int(np.nanargmax(vertex_lat))
    activation_span = float(np.nanmax(vertex_lat) - np.nanmin(vertex_lat))

    print(f"Earliest activation  : {np.nanmin(vertex_lat):.0f} ms")
    print(f"Latest activation    : {np.nanmax(vertex_lat):.0f} ms")
    print(f"Total activation     : {activation_span:.0f} ms")
    print(f"Spread across points : {np.nanmax(lat) - np.nanmin(lat):.0f} ms")
    return activation_span, earliest_vertex, latest_vertex


@app.cell
def _(
    earliest_vertex,
    epkit,
    go,
    latest_vertex,
    points,
    triangles,
    vertex_lat,
):
    # DERIVED. The activation map, with its extremes marked.
    _activation = epkit.mesh_to_plotly(
        points, triangles, scalars=vertex_lat, colorscale="Turbo_r",
        colorbar_title="LAT (ms)", name="activation",
    )
    _extremes = go.Scatter3d(
        x=points[[earliest_vertex, latest_vertex], 0],
        y=points[[earliest_vertex, latest_vertex], 1],
        z=points[[earliest_vertex, latest_vertex], 2],
        mode="markers+text",
        marker=dict(size=9, color=["red", "blue"]),
        text=["earliest", "latest"], textposition="top center", name="extremes",
    )
    epkit.plotly_figure(
        [_activation, _extremes], title="Activation map — red early, blue late"
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### How to read it — the central diagnostic question

    Two patterns, and telling them apart is most of what an ablation procedure is
    deciding.

    **Focal.** Activation radiates outward from one early site, like a stone dropped
    in water. Red in the middle, blue at the edges, no discontinuity. The earliest
    point is the source, and it is the ablation target: destroy that spot and the
    arrhythmia stops.

    **Macro-reentrant.** The earliest and latest regions sit *next to each other*,
    with the colour sweeping through the entire spectrum in between — red pressed up
    against blue along a line. Early meets late because the wave never stops: it
    circulates. There is no source to destroy. The target is the narrow **isthmus**
    the circuit has to squeeze through, and the therapy is a line of lesions across
    it.
    """
            ),
            plain(
                "Colours radiating outward from a spot: something is firing there. "
                "Red touching blue along a seam: the wave is going round in a "
                "circle, and the seam is where it closes the loop."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np):
    # DERIVED. How fast is the rhythm, and how regular?
    # Autocorrelation of each recording finds the interval between beats.
    def cycle_length(trace, minimum=250, maximum=1200):
        centred = trace - trace.mean()
        autocorrelation = np.correlate(centred, centred, "full")[len(centred) - 1 :]
        window = autocorrelation[minimum:maximum]
        return minimum + int(np.argmax(window))

    cycle_lengths = np.array([cycle_length(bundle.ecg[i]) for i in range(300)])

    print(f"median cycle length : {np.median(cycle_lengths):.0f} ms")
    print(
        f"spread              : "
        f"{np.percentile(cycle_lengths, 5):.0f} to "
        f"{np.percentile(cycle_lengths, 95):.0f} ms (5th-95th percentile)"
    )
    print(f"rate                : {60000 / np.median(cycle_lengths):.0f} beats/min")
    return (cycle_lengths,)


@app.cell(hide_code=True)
def _(activation_span, cycle_lengths, mo, np, takeaway):
    mo.vstack(
        [
            mo.md(
                rf"""
    Two numbers now settle the question for this chamber.

    **Activation span: {activation_span:.0f} ms. Cycle length:
    {np.median(cycle_lengths):.0f} ms.** The wave takes about
    {activation_span / np.median(cycle_lengths) * 100:.0f}% of the cycle to cross
    the whole atrium, which means the chamber is electrically **silent for the other
    {100 - activation_span / np.median(cycle_lengths) * 100:.0f}%**. A reentrant
    circuit cannot be silent — by definition the wavefront is somewhere in the
    circuit at every instant, so total activation time approaches the full cycle
    length. **This is not macro-reentry in this chamber.**

    A second observation falls out for free, and it is not one the file tells you:
    the cycle length is constant to within a millisecond, both between beats inside
    a single 2.5-second recording and across recordings taken many minutes apart.
    A heart in its own rhythm wanders by tens of milliseconds with breathing alone.
    A metronome that regular means the heart was being **paced** at a fixed rate
    while this map was collected — a routine thing to do, because it makes every
    point comparable.
    """
            ),
            takeaway(
                "Total activation time versus cycle length is the numerical version "
                "of the focal-or-reentrant question, and it can be answered without "
                "looking at the picture at all."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    isochrone_step = mo.ui.slider(
        2, 30, step=2, value=10, label="isochrone band width (ms)",
        show_value=True, full_width=True, debounce=True,
    )
    mo.vstack(
        [
            mo.md(
                r"""
    ### Isochrones: making crowding visible

    Rounding the timing into bands turns a smooth gradient into contour lines, in
    exactly the way a topographic map does. **Where the bands crowd together, the
    wave is moving slowly** — the same distance is taking more time. Those crowded
    regions are candidate arrhythmia substrate, and Part 7 measures them properly.
    """
            ),
            isochrone_step,
        ]
    )
    return (isochrone_step,)


@app.cell
def _(epkit, isochrone_step, np, points, triangles, vertex_lat):
    # DERIVED. Isochrones: the activation field, rounded into bands.
    _banded = np.floor(vertex_lat / isochrone_step.value) * isochrone_step.value

    epkit.plotly_figure(
        [
            epkit.mesh_to_plotly(
                points, triangles, scalars=_banded, colorscale="Turbo_r",
                colorbar_title=f"LAT ({isochrone_step.value} ms bands)",
            )
        ],
        title=f"Isochrone map — {isochrone_step.value} ms bands. "
        "Crowded bands mean slow conduction.",
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Part 6 — Voltage: mapping the tissue itself

    Timing tells you how the wave moved. **Amplitude tells you what the tissue is.**

    Healthy muscle depolarises in bulk and in step, and produces a large, sharp
    bipolar deflection. Scar has fewer surviving muscle cells under the electrode,
    so it generates less signal. The peak-to-peak height of the bipolar electrogram
    is therefore used as a proxy for how much viable muscle is there — the
    **substrate**.

    The clinical thresholds for atrial tissue are well established:

    | bipolar peak-to-peak voltage | interpretation |
    |:---|:---|
    | < 0.5 mV | dense scar |
    | 0.5 – 1.5 mV | border zone, diseased but not dead |
    | > 1.5 mV | healthy myocardium |

    These are **conventions**, calibrated against imaging and outcomes, not laws of
    physics. Different labs use slightly different numbers, the numbers differ
    between atrium and ventricle, and — as the interactive below makes obvious —
    moving them moves the diagnosis.
    """
            ),
            plain(
                "A loud signal means live muscle under the electrode; a quiet one "
                "means scar. Someone drew the line between loud and quiet at 0.5 and "
                "1.5 millivolts, and everyone uses those numbers."
            ),
        ]
    )
    return


@app.cell
def _(bipolar, bundle, n_points, n_samples, np, ref_annot, woi):
    # DERIVED, cross-checked against the system's stored values.
    our_voltage = np.full(n_points, np.nan)
    for _i in range(n_points):
        _start = int(ref_annot[_i] + woi[_i, 0])
        _stop = int(ref_annot[_i] + woi[_i, 1])
        _start, _stop = max(_start, 0), min(_stop, n_samples)
        if _stop - _start < 2:
            continue
        _segment = bipolar[_i, _start:_stop]
        our_voltage[_i] = np.nanmax(_segment) - np.nanmin(_segment)

    stored_voltage = bundle.stored_voltage.astype(float)
    _comparable = np.isfinite(our_voltage) & np.isfinite(stored_voltage)

    print(f"our peak-to-peak    : {np.nanmedian(our_voltage):.3f} mV median")
    print(f"the system's value  : {np.nanmedian(stored_voltage):.3f} mV median")
    print(
        f"correlation         : "
        f"{np.corrcoef(our_voltage[_comparable], stored_voltage[_comparable])[0, 1]:.6f}"
    )
    print(
        f"median |difference| : "
        f"{np.nanmedian(np.abs(our_voltage[_comparable] - stored_voltage[_comparable])):.2e} mV"
    )
    return (stored_voltage,)


@app.cell(hide_code=True)
def _(mo, takeaway):
    mo.vstack(
        [
            mo.md(
                r"""
    Worth pausing on, because it is the opposite of what happened in Part 2. There,
    a textbook identity between channels failed. Here, an undocumented clinical
    measurement is reproduced **exactly** — same numbers to eight decimal places —
    by the obvious one-line rule: take the biggest minus the smallest value inside
    the window of interest.
    """
            ),
            takeaway(
                "Not every claim about a file fails when you test it. The point of "
                "testing is not scepticism for its own sake; it is finding out which "
                "of your assumptions are load-bearing. Two tests, two opposite "
                "answers, both useful."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    scar_threshold = mo.ui.slider(
        0.1, 1.0, step=0.05, value=0.5, label="dense-scar threshold (mV)",
        show_value=True, full_width=True, debounce=True,
    )
    healthy_threshold = mo.ui.slider(
        1.0, 3.0, step=0.1, value=1.5, label="healthy threshold (mV)",
        show_value=True, full_width=True, debounce=True,
    )
    mo.vstack(
        [
            mo.md(
                "**The thresholds are yours to move.** Watch how much scar this "
                "patient has as you do."
            ),
            scar_threshold,
            healthy_threshold,
        ]
    )
    return healthy_threshold, scar_threshold


@app.cell
def _(
    areas,
    epkit,
    healthy_threshold,
    interpolate_to_surface,
    mo,
    np,
    points,
    scar_threshold,
    stored_voltage,
    surface_points,
    triangles,
):
    # DERIVED. Interpolate voltage onto the shell, then classify by area.
    vertex_voltage = interpolate_to_surface(surface_points, stored_voltage, points)

    vertex_area = np.zeros(len(points))
    np.add.at(vertex_area, triangles.ravel(), np.repeat(areas / 3.0, 3))

    _mapped = np.isfinite(vertex_voltage)
    _total = vertex_area[_mapped].sum()
    _low, _high = scar_threshold.value, healthy_threshold.value

    _rows = []
    for _label, _mask in [
        (f"dense scar (< {_low:.2f} mV)", _mapped & (vertex_voltage < _low)),
        (
            f"border ({_low:.2f}–{_high:.2f} mV)",
            _mapped & (vertex_voltage >= _low) & (vertex_voltage < _high),
        ),
        (f"healthy (> {_high:.2f} mV)", _mapped & (vertex_voltage >= _high)),
    ]:
        _area = vertex_area[_mask].sum()
        _rows.append(
            f"| {_label} | {int(_mask.sum()):,} | {_area:,.0f} | "
            f"{_area / _total * 100:.1f}% |"
        )

    mo.vstack(
        [
            mo.md(
                "| category | vertices | area (mm²) | share of mapped surface |\n"
                "|:---|---:|---:|---:|\n" + "\n".join(_rows) + "\n\n"
                f"*{int((~_mapped).sum()):,} vertices are unmapped and excluded "
                f"entirely.*"
            ),
            epkit.plotly_figure(
                [
                    epkit.mesh_to_plotly(
                        points, triangles,
                        scalars=np.clip(vertex_voltage, 0, 5.0),
                        colorscale=epkit.voltage_colorscale(_low, _high, 5.0),
                        cmin=0.0, cmax=5.0,
                        colorbar_title="bipolar (mV)", name="voltage",
                    )
                ],
                title=f"Voltage map — red below {_low:.2f} mV (scar), "
                f"purple above {_high:.2f} mV (healthy)",
            ),
        ]
    )
    return (vertex_voltage,)


@app.cell(hide_code=True)
def _(mo, takeaway, trap):
    mo.vstack(
        [
            mo.md(
                r"""
    At the conventional 0.5 mV, this atrium has a small amount of dense scar and a
    lot of healthy tissue. Slide the scar threshold to 1.0 mV — a value some labs
    genuinely use — and the scar burden multiplies, without a single measurement
    changing.
    """
            ),
            trap(
                "The blank region is excluded from these percentages entirely. "
                "Reporting 'X% scar' from a map that could not colour a quarter of "
                "the chamber quietly assumes the unmapped part looks like the mapped "
                "part. It usually does not: regions that are hard to reach with a "
                "catheter are also regions that are hard to ablate, and often the "
                "ones that matter."
            ),
            takeaway(
                "A substrate map is a measurement passed through two choices — an "
                "interpolation radius and a pair of thresholds — before anyone reads "
                "a diagnosis off it."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### The catch you already saw in Part 1

    A bipolar signal is the difference between two electrodes — a spatial derivative
    along the axis joining them. In the toy model, turning the wave to 90° made the
    bipolar signal vanish while the tissue stayed exactly as healthy as before.

    So some low-voltage regions on any bipolar map are not scar; they are geometry.
    The question is how much of a problem that is in practice, and the file lets us
    answer it rather than repeat the warning.

    Testing it needs two vectors: the **bipole axis** — the line joining the two
    electrodes — and the **wavefront direction** at that spot. The first is
    available now, because the file stores the position of both unipolar electrodes
    for every point.
    """
            ),
            plain(
                "We can check whether tissue looks quieter when the wave happened to "
                "run across the electrode pair rather than along it — and we can do "
                "it with the real recordings, not a model."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np):
    # MEASURED. The bipole axis, recovered from the two unipolar electrode positions.
    electrode_positions = bundle.electrode_positions.astype(float)  # (n, 3, 2)
    bipole_axis = electrode_positions[:, :, 0] - electrode_positions[:, :, 1]
    electrode_separation = np.linalg.norm(bipole_axis, axis=1)

    print(
        f"electrode separation: median {np.nanmedian(electrode_separation):.2f} mm "
        f"(p5 {np.nanpercentile(electrode_separation, 5):.2f}, "
        f"p95 {np.nanpercentile(electrode_separation, 95):.2f})"
    )
    print("consistent with the 2 mm spacing of a 20-pole mapping catheter.")

    bipole_axis = bipole_axis / np.linalg.norm(bipole_axis, axis=1, keepdims=True)
    return (bipole_axis,)


@app.cell(hide_code=True)
def _(mo, trap):
    mo.vstack(
        [
            mo.md(
                r"""
    The second vector — wavefront direction — is exactly what conduction velocity
    gives us, so the test has to wait for Part 7.
    """
            ),
            trap(
                "**A tempting shortcut that is wrong.** The MATLAB file has a field "
                "called `barDirection`, and the name invites you to read it as the "
                "electrode orientation. It is not. OpenEP loads it as "
                "`electric.surface.normals` — the *surface normal* at each point, "
                "the direction perpendicular to the wall. Test one against the other "
                "and the alignment is exactly 1.000 at every point, because they are "
                "the same array. **A field name is not a specification.**"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Part 7 — Conduction velocity: how fast, and which way

    We have a timing value at every vertex of a surface. Its **gradient** — the
    direction of steepest increase, and how steep — tells us how fast and in which
    direction the wave moved.

    If activation time is $t(\mathbf{x})$ in milliseconds and position is in
    millimetres, then $|\nabla t|$ has units of ms/mm: the time cost per millimetre
    travelled. Speed is its reciprocal,

    $$\mathrm{CV} = \frac{1}{|\nabla t|},$$

    and the wave travels along $+\nabla t$, toward later activation.

    On a triangulated surface the gradient is computed one triangle at a time. A
    linear field over a triangle has a constant gradient, obtained by solving the
    two edge equations together with the constraint that the gradient lies in the
    plane of the triangle. Vertex values are then the average of the triangles
    meeting there.
    """
            ),
            plain(
                "Contour lines on a hiking map: where they bunch up the ground is "
                "steep, and downhill is perpendicular to them. Same arithmetic here, "
                "with time in place of altitude — bunched contours mean the wave is "
                "crawling."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking the tool before trusting the tool

    `openep-py` has **no conduction-velocity function at all**, so unlike everywhere
    else in this notebook there is no reference implementation to check against.

    When there is nothing to compare with, build a case where the answer is known.
    Paint a perfectly planar wave travelling at exactly 1 mm/ms across a flat
    triangulated sheet, and require the code to give back 1000 mm/s. (This is the
    same check the test suite runs, in `tests/test_analysis.py`.)
    """)
    return


@app.cell
def _(epkit, np):
    # DERIVED. A synthetic planar wave with a known answer.
    def flat_sheet(n=21, spacing=1.0):
        """A flat triangulated sheet in the z = 0 plane, n x n vertices."""
        _xs = np.arange(n) * spacing
        _xx, _yy = np.meshgrid(_xs, _xs, indexing="ij")
        _sheet = np.column_stack([_xx.ravel(), _yy.ravel(), np.zeros(n * n)])
        _tris = []
        for _i in range(n - 1):
            for _j in range(n - 1):
                _a, _b = _i * n + _j, _i * n + _j + 1
                _c, _d = (_i + 1) * n + _j, (_i + 1) * n + _j + 1
                _tris += [[_a, _b, _d], [_a, _d, _c]]
        return _sheet, np.array(_tris)

    _sheet_points, _sheet_triangles = flat_sheet()
    _synthetic_lat = _sheet_points[:, 0]  # 1 ms per mm travelled in +x

    _speed, _dir = epkit.conduction_velocity(
        _sheet_points, _sheet_triangles, _synthetic_lat
    )

    print("planar wave at a known 1000 mm/s, recovered:")
    print(f"  median speed     : {np.nanmedian(_speed):,.3f} mm/s")
    print(f"  worst error      : {np.nanmax(np.abs(_speed - 1000.0)):.2e} mm/s")
    print(f"  median direction : {np.nanmedian(_dir, axis=0)}  (should be +x)")
    return


@app.cell(hide_code=True)
def _(deeper, mo, takeaway):
    mo.vstack(
        [
            mo.md(
                r"""
    Exact to numerical precision. The implementation is sound; whatever goes wrong
    from here is the *data*, not the arithmetic. That distinction is worth being
    able to make.

    Normal atrial conduction is roughly **600–1200 mm/s**. Below about 300 mm/s is
    slow conduction — the substrate that sustains reentry.
    """
            ),
            deeper(
                "why the test sheet has to be flat",
                r"""
    Try the same test on the real atrial mesh — set the activation time of each
    vertex to a linear function of its 3D position — and the recovered speed is not
    1000 mm/s. It is a median of about 1144, with a long tail.

    Nothing is broken. A field that is planar in 3D space does not describe a wave
    travelling at constant speed *along a curved surface*: where the wall tilts away
    from the direction of travel, the wave has to cover more surface distance per
    millisecond, so the along-surface speed genuinely is higher.

    That is not just a testing nuisance. It is a real effect on real hearts: surface
    conduction velocity is measured along a curved wall, and curvature that the mesh
    resolves poorly shows up as speed the tissue never had.
    """,
            ),
            takeaway(
                "When there is no reference implementation to check against, "
                "manufacture a case whose answer you know. It is the cheapest test "
                "in existence and it separates 'my code is wrong' from 'my data is "
                "hard'."
            ),
        ]
    )
    return


@app.cell
def _(epkit, np, plt, points, triangles, vertex_lat):
    # DERIVED. Conduction velocity from the interpolated activation field.
    cv_speed, cv_direction = epkit.conduction_velocity(points, triangles, vertex_lat)

    _finite = cv_speed[np.isfinite(cv_speed)]
    print(f"vertices with a CV value : {_finite.size:,}")
    print(f"median CV                : {np.median(_finite):,.0f} mm/s")
    print(
        f"IQR                      : {np.percentile(_finite, 25):,.0f} to "
        f"{np.percentile(_finite, 75):,.0f} mm/s"
    )
    print(f"slow (< 300 mm/s)        : {np.mean(_finite < 300) * 100:.1f}% of mapped surface")
    print(f"implausible (> 2000 mm/s): {np.mean(_finite > 2000) * 100:.1f}%")

    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.hist(np.clip(_finite, 0, 3000), bins=60, color="#55a868", edgecolor="white", lw=0.4)
    _ax.axvline(300, color="crimson", ls="--", label="300 mm/s — slow conduction")
    _ax.set_xlabel("conduction velocity (mm/s)")
    _ax.set_ylabel("vertices")
    _ax.set_title("Conduction velocity distribution")
    _ax.legend()
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return cv_direction, cv_speed


@app.cell
def _(cv_direction, cv_speed, epkit, go, np, points, triangles):
    # DERIVED. The CV map, with propagation arrows on top.
    _cv_map = epkit.mesh_to_plotly(
        points, triangles, scalars=np.clip(cv_speed, 0, 2000), colorscale="Viridis",
        cmin=0, cmax=2000, colorbar_title="CV (mm/s)", name="conduction velocity",
    )

    _step = max(len(points) // 400, 1)
    _sample = np.arange(0, len(points), _step)
    _sample = _sample[np.isfinite(cv_direction[_sample]).all(axis=1)]

    arrows = go.Cone(
        x=points[_sample, 0], y=points[_sample, 1], z=points[_sample, 2],
        u=cv_direction[_sample, 0] * 3.0,
        v=cv_direction[_sample, 1] * 3.0,
        w=cv_direction[_sample, 2] * 3.0,
        sizemode="absolute", sizeref=4.0, showscale=False,
        colorscale=[[0, "black"], [1, "black"]], name="propagation",
    )

    epkit.plotly_figure(
        [_cv_map, arrows], title="Conduction velocity, with propagation direction"
    )
    return (arrows,)


@app.cell(hide_code=True)
def _(cv_speed, mo, np, trap):
    mo.vstack(
        [
            mo.md(
                rf"""
    ### When conduction velocity lies

    CV is the most derived quantity in this notebook: a numerical gradient, of an
    interpolated field, built from annotated timings, each of which we have already
    seen is a choice. Every one of those layers can inject error, and three
    situations produce confidently wrong numbers:

    - **Wave collision.** Where two wavefronts meet, the timing field has a genuine
      discontinuity — a crease. The gradient there is enormous and the computed CV
      collapses toward zero, mimicking slow conduction that does not exist.
    - **Breakthrough.** Where a wave surfaces from elsewhere in the wall (this is a
      *surface* map of a three-dimensional muscle), the surface gradient nearly
      vanishes and CV goes implausibly high. That is what the
      **{np.mean(cv_speed[np.isfinite(cv_speed)] > 2000) * 100:.0f}% above
      2000 mm/s** in the numbers above is measuring: not superhumanly fast tissue,
      but a wave that was never travelling in the plane we measured.
    - **Sparse sampling.** Where the interpolator invented the field, its gradient
      describes the interpolator, not the tissue.
    """
            ),
            trap(
                "Before believing any specific slow-conduction region, cross-check "
                "it against the coverage map from Part 3 and the voltage map from "
                "Part 6. Slow conduction in a well-sampled, low-voltage region is a "
                "finding. The same colour in a sparsely sampled region is an "
                "artefact of the interpolator."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Back to Part 6: does bipolar voltage really depend on direction?

    Now we have both vectors. For each mapping point, take the bipole axis and the
    local propagation direction and measure their alignment as $|\cos\theta|$: 1
    when the wave travels along the electrode pair, 0 when it crosses perpendicular.

    Theory — and the toy model in Part 1 — predicts systematically higher bipolar
    voltage at high alignment.
    """)
    return


@app.cell
def _(
    bipole_axis,
    cv_direction,
    np,
    points,
    spatial,
    stats,
    stored_voltage,
    surface_points,
):
    # DERIVED. Bipole-axis vs wavefront alignment, against measured voltage.
    nearest_vertex = spatial.cKDTree(points).query(surface_points, k=1)[1]
    wavefront = cv_direction[nearest_vertex]

    alignment = np.abs(np.sum(bipole_axis * wavefront, axis=1))
    _usable = np.isfinite(alignment) & np.isfinite(stored_voltage)

    _lower, _upper = np.nanpercentile(alignment[_usable], [33, 67])
    _a, _v = alignment[_usable], stored_voltage[_usable]

    print(f"usable points : {int(_usable.sum())}")
    print(f"{'group':<34}{'n':>6}{'median mV':>12}")
    alignment_medians = {}
    for _label, _key, _mask in [
        (f"perpendicular (alignment < {_lower:.2f})", "perpendicular", _a <= _lower),
        ("intermediate", "intermediate", (_a > _lower) & (_a < _upper)),
        (f"parallel (alignment > {_upper:.2f})", "parallel", _a >= _upper),
    ]:
        alignment_medians[_key] = float(np.nanmedian(_v[_mask]))
        print(f"{_label:<34}{int(_mask.sum()):>6}{alignment_medians[_key]:>12.3f}")

    alignment_rho, alignment_p = stats.spearmanr(_a, _v)
    print(f"\nSpearman rho = {alignment_rho:.3f}, p = {alignment_p:.3g}")
    return alignment_medians, alignment_p, alignment_rho


@app.cell(hide_code=True)
def _(
    alignment_medians,
    alignment_p,
    alignment_rho,
    mo,
    n_points,
    takeaway,
    trap,
):
    mo.vstack(
        [
            mo.md(
                rf"""
    The effect is **there, in the predicted direction, and small**:
    {alignment_medians["perpendicular"]:.2f} mV when the wave crosses the bipole
    side-on against {alignment_medians["parallel"]:.2f} mV when it runs along it —
    a difference of
    {(alignment_medians["parallel"] / alignment_medians["perpendicular"] - 1) * 100:.0f}%.

    Now be careful with that p-value. With {n_points:,} points, a correlation of
    $\rho = {alignment_rho:.3f}$ clears the conventional significance bar
    ($p = {alignment_p:.3f}$) while explaining well under 1% of the variance in
    voltage. **Statistically detectable and clinically meaningful are different
    claims**, and this is a clean example of the gap between them.

    So the honest conclusion is narrower than the textbook warning implies.
    Direction dependence is real and measurable here, but in this dataset it is not
    what separates a 0.3 mV scar reading from a 2 mV healthy one. Tissue is doing
    almost all of the work. The mechanism still matters — it is why unipolar
    voltage, which has no electrode axis and therefore no direction blindness, is
    read alongside bipolar, and why *omnipolar* mapping was invented to remove the
    dependence altogether — but on this evidence it is a second-order correction,
    not a reason to distrust the substrate map.
    """
            ),
            trap(
                "Two caveats keep even that modest conclusion tentative. The "
                "propagation direction is itself derived from an interpolated field, "
                "so it carries every uncertainty from Part 4. And it describes the "
                "mapped beat, whereas each point was recorded at a different moment "
                "— potentially with the wave arriving from a different direction."
            ),
            takeaway(
                "'Statistically significant' answers whether an effect exists. It "
                "says nothing about whether the effect is big enough to change what "
                "you do. Always ask for the effect size, and Part 8 shows what a "
                "large one looks like in the same dataset."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(jargon, mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Part 8 — Ablation: delivering and recording therapy

    Diagnosis identifies the target. **Ablation** destroys it.

    The catheter tip is pressed against the wall and radiofrequency current — the
    same frequency band as a radio transmitter, around 500 kHz — is passed from the
    tip into the tissue and out through a large patch on the patient's back. The
    tissue's own resistance turns that current into heat. Above roughly **50 °C**
    the muscle cells die and are replaced over the following weeks by fibrous scar,
    which conducts nothing. A well-placed line of such lesions blocks the circuit
    permanently.

    Getting that right is harder than it sounds, and the reason is contact. Too
    little pressure and the heat goes into the blood instead of the wall, producing
    a shallow lesion that looks perfect on the map today and recovers in six weeks.
    Too much and you risk perforating a wall three millimetres thick, or damaging
    the oesophagus lying directly behind the atrium.

    So modern catheters measure the **contact force** at the tip, and modern systems
    compute running quality indices from force, power and time. This export claims
    to record all of it. Before plotting any of it, we check whether the numbers are
    physically possible — the same discipline Part 2 applied to the electrograms.
    """
            ),
            plain(
                "Burning a spot of muscle to make it electrically dead. The trick is "
                "pressing hard enough for the heat to go into the wall, for long "
                "enough that the damage is permanent, without going through it."
            ),
            jargon(
                [
                    ("radiofrequency (RF)", "the alternating current used to heat tissue"),
                    ("lesion", "one burned spot"),
                    ("contact force", "how hard the catheter tip presses on the wall, in grams"),
                    ("impedance", "electrical resistance seen by the catheter tip, in ohms"),
                    ("FTI", "force–time integral: force multiplied by duration, in gram-seconds"),
                    ("Ablation Index", "a proprietary weighted combination of force, power and time"),
                    ("durability", "whether a lesion is still blocking conduction months later"),
                ]
            ),
        ]
    )
    return


@app.cell
def _(bundle, np):
    # MEASURED — and audited before use. For each channel, what does the peak
    # value have to look like for the channel to be doing its job at all?
    channels = [
        # label, samples, plausible peak range, why
        ("power (W)", bundle.ablation_power.astype(float), 5, 60),
        ("temperature (C)", bundle.ablation_temperature.astype(float), 40, 80),
        ("impedance (ohm)", bundle.ablation_impedance.astype(float), 80, 400),
        ("contact force (g)", bundle.force_values.astype(float), 5, 100),
    ]

    print(
        f"RF samples: {len(bundle.ablation_times):,}   "
        f"force samples: {len(bundle.force_values):,}\n"
    )
    print(f"{'channel':<20}{'min':>10}{'max':>12}{'plausible peak':>18}   verdict")
    for _label, _series, _low, _high in channels:
        _finite = _series[np.isfinite(_series)]
        _peak = np.nanmax(_finite)
        _ok = bool(_finite.size) and (_low <= _peak <= _high)
        print(
            f"{_label:<20}{np.nanmin(_finite):>10.1f}{_peak:>12.1f}"
            f"{f'{_low}-{_high}':>18}   {'usable' if _ok else 'NOT POPULATED'}"
        )

    _impedance = bundle.ablation_impedance.astype(float)
    print(f"\nimpedance, first 6 samples: {_impedance[:6]}")
    print(f"consecutive differences   : {np.diff(_impedance[:6])}")
    return


@app.cell(hide_code=True)
def _(bundle, mo, takeaway):
    mo.vstack(
        [
            mo.md(
                rf"""
    Three of the four channels do not survive the check.

    - **Power is identically zero** across every sample. Not low — zero. No
      radiofrequency generator delivers a lesion at 0 W.
    - **Impedance** starts at a round 2000 and rises in exact steps of 100, reaching
      44,600. That is a counter, not ohms; real catheter impedance sits near
      100–150 Ω and *falls* by a few ohms as tissue heats — which is itself one of
      the signals operators watch.
    - **Temperature** never exceeds 31 °C, below body temperature, let alone the
      ~50 °C needed to kill a myocyte.
    - **Contact force is real**: {bundle.meta["ablation_ranges"]["force"][0]:.2f} to
      {bundle.meta["ablation_ranges"]["force"][1]:.1f} g is exactly the expected
      range for a force-sensing catheter.

    The other dataset in the OpenEP collection shows the same pattern for power
    (identically zero) and impedance (a counter reaching
    {bundle.meta["ablation_ranges_other_dataset"]["impedance"][1]:,.0f}), though its
    temperature channel does reach
    {bundle.meta["ablation_ranges_other_dataset"]["temperature"][1]:.0f} °C. So this
    is a systematic property of how these files were exported, not a corrupted
    download.
    """
            ),
            takeaway(
                "Had we plotted the impedance channel without looking, we would have "
                "produced a clean rising line and narrated it as lesion formation. "
                "It would have been pure fiction, and it would have looked "
                "completely convincing. This is what 'verify, don't assume' costs, "
                "and what it buys."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np, plt):
    # MEASURED. Contact force is the one raw channel worth plotting.
    force_time = (bundle.force_times.astype(float) - bundle.force_times[0]) / 1000.0
    force_values = bundle.force_values.astype(float)

    _fig, _axes = plt.subplots(2, 1, figsize=(11, 5.5))
    _axes[0].plot(force_time / 60.0, force_values, lw=0.4, color="#4c72b0")
    _axes[0].set_xlabel("time through the procedure (minutes)")
    _axes[0].set_ylabel("contact force (g)")
    _axes[0].set_title("Catheter-tissue contact force across the whole case")
    _axes[0].grid(alpha=0.25)

    _busiest = int(np.argmax(np.convolve(force_values, np.ones(600) / 600, mode="same")))
    _window = slice(max(_busiest - 600, 0), min(_busiest + 600, len(force_values)))
    _axes[1].plot(
        force_time[_window] - force_time[_window][0], force_values[_window],
        lw=0.9, color="#c44e52",
    )
    _axes[1].axhline(10, color="grey", ls="--", label="10 g — a typical contact target")
    _axes[1].set_xlabel("time (s)")
    _axes[1].set_ylabel("force (g)")
    _axes[1].set_title("A two-minute window: contact is intermittent, not steady")
    _axes[1].legend()
    _axes[1].grid(alpha=0.25)
    plt.tight_layout()

    print(
        f"median force while in contact (>1 g) : "
        f"{np.nanmedian(force_values[force_values > 1]):.1f} g"
    )
    print(f"share of samples below 5 g           : {np.mean(force_values < 5) * 100:.1f}%")
    _fig
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    Contact is **intermittent and often light**. The catheter is inside a chamber
    that contracts once a second and moves with every breath, held by an operator
    steering from a metre away through a vein. This trace is what that actually
    looks like, and it is precisely why force-aware lesion metrics were invented.

    ### Lesion tags: what the operator left behind

    Each completed lesion is recorded as a **tag** — a marked position with the
    metrics describing lesion quality. `openep-py` does not load these (its own
    source says so), so they are parsed straight out of the MATLAB `rfindex`
    structure by `epkit/data.py`.
    """
            ),
            plain(
                "Every burn the operator delivered is saved as a dot in 3D with a "
                "little report card attached: how long, how hard, how hot."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np):
    # MEASURED. The ablation lesion tags.
    print(f"lesion tags : {len(bundle.tag_points)}")
    print(f"index metric: {bundle.meta['index_name']}")
    print()
    print(f"{'metric':<26}{'median':>10}{'min':>10}{'max':>10}")
    for _label, _values in [
        ("duration (s)", bundle.tag_duration),
        ("average force (g)", bundle.tag_avg_force),
        ("max power (W)", bundle.tag_max_power),
        ("max temperature (C)", bundle.tag_max_temp),
        ("force-time integral (gs)", bundle.tag_fti),
        ("Ablation Index", bundle.tag_index_value),
    ]:
        _v = np.asarray(_values, dtype=float)
        print(
            f"{_label:<26}{np.nanmedian(_v):>10.1f}{np.nanmin(_v):>10.1f}"
            f"{np.nanmax(_v):>10.1f}"
        )
    return


@app.cell(hide_code=True)
def _(deeper, mo, takeaway):
    mo.vstack(
        [
            mo.md(
                r"""
    Note what just happened. The raw telemetry was mostly unusable, but the
    **per-lesion summaries are intact and physically sensible**: about 31 W of
    power, peak temperatures in the mid-30s, 8.5 g of average contact, lesions
    lasting 4 to 36 seconds. The mapping system computed and stored these at the
    time; the underlying sample streams simply were not exported. **Summary
    survived, raw did not.**

    Two of these deserve unpacking, because they are what an operator is watching
    while burning:

    - **Force–time integral (FTI)** is contact force multiplied by duration, in
      gram-seconds. It exists because lesion depth depends on how hard the catheter
      pressed and for how long, not merely on how much power was delivered.
    - **Ablation Index** goes further, combining force, power and time in a weighted
      formula fitted to observed lesion depth. It is the number many operators watch
      in real time, with targets that differ by location — commonly around 350–400
      on the thin posterior wall, higher anteriorly where the muscle is thicker.
    """
            ),
            deeper(
                "why anyone bothers with a lesion-quality index",
                r"""
    Because durability is the central failure mode of atrial fibrillation ablation.

    A lesion that stuns tissue without killing it produces a perfect-looking map at
    the end of the procedure, and then recovers. Weeks later the pulmonary vein
    reconnects through the gap, the arrhythmia returns, and the patient needs a
    repeat procedure — which is why a substantial fraction of AF ablation patients
    have a second one, and why reconnected gaps are what the second operator
    usually finds.

    Nobody can see lesion depth during a procedure. Indices like Ablation Index are
    an attempt to predict it from what *can* be measured, and they are validated
    against exactly that outcome: does the line still block, months later.
    """,
            ),
            takeaway(
                "When raw telemetry is missing, look for the derived summaries — and "
                "when the summaries are present, ask which they are: computed at the "
                "time by an instrument, or reconstructed later by you."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np, plt, stats):
    # MEASURED. Ablation Index against its components.
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for _ax, (_label, _values) in zip(
        _axes,
        [
            ("Ablation Index", bundle.tag_index_value),
            ("Force-time integral (gs)", bundle.tag_fti),
            ("Average force (g)", bundle.tag_avg_force),
        ],
    ):
        _v = np.asarray(_values, dtype=float)
        _ax.hist(_v[np.isfinite(_v)], bins=20, color="#937860", edgecolor="white", lw=0.5)
        _ax.set_title(_label)
        _ax.set_ylabel("lesions")
        _ax.grid(alpha=0.25)
    plt.tight_layout()

    _usable = np.isfinite(bundle.tag_index_value) & np.isfinite(bundle.tag_fti)
    _rho, _ = stats.spearmanr(
        bundle.tag_index_value[_usable].astype(float),
        bundle.tag_fti[_usable].astype(float),
    )
    print(f"Ablation Index vs force-time integral, Spearman rho = {_rho:.3f}")
    print("Related but not identical — the index weights power and time differently.")
    _fig
    return


@app.cell
def _(bundle, epkit, go, np, points, triangles):
    # MEASURED. Lesion tags on the anatomy, coloured by force-time integral.
    _shell = epkit.mesh_to_plotly(
        points, triangles, name="shell", opacity=0.55, showscale=False
    )
    _fti = np.asarray(bundle.tag_fti, dtype=float)

    _lesions = go.Scatter3d(
        x=bundle.tag_points[:, 0],
        y=bundle.tag_points[:, 1],
        z=bundle.tag_points[:, 2],
        mode="markers",
        marker=dict(
            size=7, color=_fti, colorscale="Inferno", showscale=True,
            colorbar=dict(title="FTI (gs)", thickness=14, len=0.7),
            line=dict(width=0.5, color="white"),
        ),
        text=[
            f"lesion {_i}: {_f:.0f} gs, {_d:.1f} s, {_p:.0f} W"
            for _i, (_f, _d, _p) in enumerate(
                zip(_fti, bundle.tag_duration, bundle.tag_max_power)
            )
        ],
        hoverinfo="text",
        name="lesions",
    )

    epkit.plotly_figure(
        [_shell, _lesions],
        title=f"{len(bundle.tag_points)} ablation lesions, coloured by force-time integral",
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    Rotate that until you are looking at the back of the chamber, and the two rings
    from the schematic in Part 0 are there in real coordinates: the operator drew a
    closed loop around each pair of pulmonary veins. That is a pulmonary vein
    isolation, recorded lesion by lesion.

    ### Three different things are called "tags"

    | kind | what it marks | in this file |
    |:---|:---|:---|
    | **ablation tags** | where therapy was delivered | **90, with full metrics** |
    | **anatomical / landmark tags** | vein openings, valve, phrenic nerve | the field exists |
    | **point tags** | electrogram features: fractionated, double potential, late potential | field exists, **empty** |

    That last row is the honest gap in this dataset, and it is worth confirming
    rather than asserting.
    """
            ),
            plain(
                "The operator's own clinical annotations of individual recordings — "
                "the 'this one looks abnormal' marks — did not survive the export. "
                "So we will detect those features ourselves."
            ),
        ]
    )
    return


@app.cell
def _(bundle):
    # Confirming the gap rather than asserting it.
    _entries = list(bundle.point_tags)
    _non_empty = [t for t in _entries if t.strip() not in ("", "[]")]

    print(f"tag entries in the raw file  : {len(_entries)}")
    print(f"non-empty tags               : {len(_non_empty)}")
    print(f"point labels that DID survive: {list(bundle.internal_names[:6])} ...")
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    One entry per point, every one empty — and `openep-py` does not even expose the
    field. The point labels (`P9`, `P10`, …) survived; the clinical annotations did
    not.

    ### Detecting fractionation ourselves

    A **fractionated** electrogram shows many small deflections instead of one sharp
    spike. Physically it means the wave is picking its way through surviving strands
    of muscle threaded between patches of scar, each strand activating at a slightly
    different moment, so the electrode hears a stutter rather than a snap. Those
    regions are ablation targets in their own right, and they are exactly where the
    annotation rules in Part 2 disagreed.

    Counting deflections is a two-line operation: differentiate, normalise, and count
    how many times the signal rises past a fraction of its own peak slope.
    """
            ),
            plain(
                "A crisp click means healthy muscle activating together. A stutter "
                "means the signal is threading through scar. We can count the "
                "stutters."
            ),
        ]
    )
    return


@app.cell
def _(bipolar, epkit, fs, np, plt, ref_annot, times_ms, woi):
    # DERIVED. Detect fractionation, since the operator's tags are absent.
    window_samples = (
        int(ref_annot[0] + woi[0, 0]),
        int(ref_annot[0] + woi[0, 1]),
    )
    deflections = epkit.fractionation_index(bipolar, fs, window_samples)

    fractionation_threshold = np.nanpercentile(deflections, 90)
    complex_points = deflections >= fractionation_threshold

    print(
        f"deflection count : median {np.nanmedian(deflections):.0f}, "
        f"90th percentile {fractionation_threshold:.0f}"
    )
    print(
        f"flagged as complex: {int(complex_points.sum())} of "
        f"{len(deflections)} points"
    )

    _simple = int(np.nanargmin(np.where(np.isfinite(deflections), deflections, np.inf)))
    _complex = int(np.nanargmax(np.where(np.isfinite(deflections), deflections, -np.inf)))

    _fig, _axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    _slice = slice(*window_samples)
    _axes[0].plot(times_ms[_slice], bipolar[_simple, _slice], color="#2ca02c", lw=1.2)
    _axes[0].set_title(f"sharp / organised — {deflections[_simple]:.0f} deflections")
    _axes[1].plot(times_ms[_slice], bipolar[_complex, _slice], color="#d62728", lw=1.2)
    _axes[1].set_title(f"fractionated — {deflections[_complex]:.0f} deflections")
    _axes[1].set_xlabel("time (ms)")
    for _ax in _axes:
        _ax.set_ylabel("mV")
        _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return complex_points, deflections, window_samples


@app.cell
def _(complex_points, deflections, np, stats, stored_voltage):
    # DERIVED. Do our complex points agree with low voltage, as theory predicts?
    comparable_points = np.isfinite(deflections) & np.isfinite(stored_voltage)

    voltage_when_complex = float(
        np.nanmedian(stored_voltage[complex_points & comparable_points])
    )
    voltage_when_simple = float(
        np.nanmedian(stored_voltage[~complex_points & comparable_points])
    )
    fractionation_rho, fractionation_p = stats.spearmanr(
        deflections[comparable_points], stored_voltage[comparable_points]
    )

    print(f"median voltage, flagged complex : {voltage_when_complex:.3f} mV")
    print(f"median voltage, everything else : {voltage_when_simple:.3f} mV")
    print(
        f"Spearman rho (deflections vs voltage) = {fractionation_rho:.3f}, "
        f"p = {fractionation_p:.3g}"
    )
    return fractionation_rho, voltage_when_complex, voltage_when_simple


@app.cell(hide_code=True)
def _(
    alignment_rho,
    fractionation_rho,
    mo,
    n_points,
    takeaway,
    voltage_when_complex,
    voltage_when_simple,
):
    mo.vstack(
        [
            mo.md(
                rf"""
    This is the strongest validation in the notebook, and it came for free.

    Nothing in the fractionation detector knows anything about voltage — it counts
    deflections in a waveform, and nothing else. Yet the points it flags have a
    median voltage of **{voltage_when_complex:.2f} mV against
    {voltage_when_simple:.2f} mV** for everything else, landing almost exactly on
    the 0.5 mV dense-scar threshold from Part 6, with a correlation of
    $\rho = {fractionation_rho:.2f}$ across {n_points:,} points.

    Two independent measurements of tissue health — *how fragmented is the signal*
    and *how large is the signal* — agree. That is what a real physiological
    relationship looks like, and it is worth putting side by side with the
    direction-dependence result from Part 7:

    | claim | correlation | verdict |
    |:---|:---|:---|
    | fragmented signal ↔ low voltage | $\rho = {fractionation_rho:.2f}$ | large, mechanistic, believable |
    | wave direction ↔ bipolar voltage | $\rho = {alignment_rho:.2f}$ | statistically significant, practically negligible |

    Same test, same dataset, effect sizes an order of magnitude apart. Both had
    p-values below 0.05.
    """
            ),
            takeaway(
                "A p-value tells you whether an effect is distinguishable from zero. "
                "Only the effect size tells you whether it matters. Publishing the "
                "first without the second is how a literature fills up with real, "
                "reproducible, useless findings."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The promised callback to Part 2

    Part 2 found that our annotation rule matched the system's for about 92% of
    points, and disagreed by as much as 81 ms for the rest. The claim was that those
    disagreements are not random. Now we can test it: **do the annotation rules
    diverge where the electrogram is fragmented?**
    """)
    return


@app.cell
def _(annotation_pairs, deflections, lat, np, our_lat, stats):
    # DERIVED. Where do the two annotation rules disagree?
    _gap = np.abs(our_lat - lat)
    _usable = annotation_pairs & np.isfinite(deflections)

    _rho, _p = stats.spearmanr(_gap[_usable], deflections[_usable])
    _simple = _usable & (deflections <= 2)
    _fragmented = _usable & (deflections >= 5)

    print(f"points compared : {int(_usable.sum())}")
    print(f"{'electrogram':<28}{'n':>6}{'median gap':>13}{'gap > 5 ms':>13}")
    for _label, _mask in [
        ("simple (<=2 deflections)", _simple),
        ("fragmented (>=5)", _fragmented),
    ]:
        print(
            f"{_label:<28}{int(_mask.sum()):>6}"
            f"{np.median(_gap[_mask]):>11.0f} ms"
            f"{np.mean(_gap[_mask] > 5) * 100:>12.0f}%"
        )
    print(f"\nSpearman rho (annotation gap vs deflection count) = {_rho:.3f}, p = {_p:.2g}")
    return


@app.cell(hide_code=True)
def _(mo, takeaway):
    mo.vstack(
        [
            mo.md(
                r"""
    Exactly as predicted, and the size of the split is striking. On simple
    electrograms the two rules essentially never disagree by more than 5 ms. On
    fragmented ones, a third of points do.

    This is the mechanism behind a claim that gets made loosely all the time —
    "annotation is subjective in diseased tissue". It is not subjective everywhere.
    It is *deterministic* where the signal is clean and *rule-dependent* precisely
    where the tissue is abnormal, which is precisely where the clinical decision is
    being made.
    """
            ),
            takeaway(
                "Uncertainty in this data is not spread evenly. It concentrates in "
                "the regions that matter most — poorly sampled, low voltage, "
                "fragmented — which is exactly where a smooth colourful map hides it "
                "best."
            ),
        ]
    )
    return


@app.cell
def _(
    complex_points,
    epkit,
    go,
    np,
    points,
    surface_points,
    triangles,
    vertex_voltage,
):
    # DERIVED. Our own complex-point tags on the anatomy.
    _flagged = complex_points & np.isfinite(surface_points).all(axis=1)

    _voltage_shell = epkit.mesh_to_plotly(
        points, triangles, scalars=np.clip(vertex_voltage, 0, 5.0),
        colorscale=epkit.voltage_colorscale(), cmin=0, cmax=5.0,
        colorbar_title="bipolar (mV)", opacity=0.9,
    )
    _flags = go.Scatter3d(
        x=surface_points[_flagged, 0],
        y=surface_points[_flagged, 1],
        z=surface_points[_flagged, 2],
        mode="markers",
        marker=dict(size=5, color="black", symbol="x"),
        name="derived complex points",
    )
    epkit.plotly_figure(
        [_voltage_shell, _flags],
        title="Derived complex-point tags over the voltage map",
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ## Part 9 — Reading the procedure out of the data

    Everything so far has treated the file as measurements. This part treats it as
    **evidence about how it was collected** — which turns out to be recoverable, and
    which changes how the measurements should be read.

    ### Point-by-point or continuous?

    There are two ways to collect the electrical points.

    **Point-by-point.** Position the catheter, confirm contact and stability, accept
    one point, move on. Slow, deliberate, every point reviewed by a human. Hundreds
    of points in a case.

    **Continuous / multi-electrode.** A catheter carrying many electrodes sweeps
    around while the system automatically accepts everything that meets its quality
    criteria. Thousands to tens of thousands of points, nobody reviewing each one.

    The acquisition mode is not stored anywhere in the export. It leaves
    fingerprints anyway.
    """
            ),
            plain(
                "Did a human accept each of these recordings one at a time, or did a "
                "machine hoover them up ten at a time? Nobody wrote it down, but the "
                "data remembers."
            ),
        ]
    )
    return


@app.cell
def _(bundle, n_points, np):
    # DERIVED. Fingerprints of how the data was collected.
    _labels = list(bundle.electrode_names)
    unique_labels, label_counts = np.unique(_labels, return_counts=True)

    print("electrode channels contributing points:")
    for _label, _count in sorted(
        zip(unique_labels, label_counts), key=lambda pair: -pair[1]
    ):
        print(f"  {_label:<14}{_count:>6} points")

    print(f"\ndistinct electrode pairs : {len(unique_labels)}")
    print(f"points per pair          : {np.median(label_counts):.0f} median")
    print(f"total points             : {n_points}")
    return (unique_labels,)


@app.cell(hide_code=True)
def _(mo, takeaway, unique_labels):
    mo.vstack(
        [
            mo.md(
                rf"""
    {len(unique_labels)} bipolar channels named `20A_1-2` through `20A_19-20`. Read
    the naming: electrodes 1 through 20 on a catheter called `20A`, paired up
    consecutively. This is a **20-pole circular mapping catheter** — a loop of
    twenty electrodes that sits against the wall and records from ten pairs at once.

    With the points spread fairly evenly across all ten channels, this was
    multi-electrode acquisition, not one-point-at-a-time. **The acquisition mode has
    been recovered from electrode naming alone.**

    It also explains the clustering found in Part 3: ten electrodes firing together
    drop ten points within a couple of millimetres of each other, which is exactly
    why nearest-neighbour spacing looked so much better than actual surface
    coverage. Two findings, three parts apart, with a single cause.
    """
            ),
            takeaway(
                "Metadata is rarely as absent as it looks. Channel names, array "
                "shapes, sentinel values and sampling rates are all evidence about "
                "the instrument, and reading them changes how much you trust what "
                "the instrument produced."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, plain):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Pace mapping, and a method the data cannot support

    Sometimes activation mapping cannot find the source — commonly because the
    arrhythmia is infrequent, or will not stay running long enough to map. Then the
    logic gets inverted.

    Instead of recording where the wave went, **pace from a candidate site and see
    whether the resulting body-surface ECG matches the clinical arrhythmia**. If
    pacing from a spot reproduces the arrhythmia's 12-lead pattern, then that spot
    activates the heart the same way the arrhythmia does — so it is at or very near
    the source.

    **PASO** is CARTO's module for this. It correlates the paced 12-lead complex
    against a stored template and reports a percentage, with >95% across all twelve
    leads conventionally counting as a good match.

    **This dataset cannot reproduce PASO**, and it is worth showing why rather than
    quietly skipping it. The file stores a *single* ECG channel, not twelve, and no
    paced templates. What follows demonstrates the correlation mathematics on the
    one lead available. It is the method, not the tool.
    """
            ),
            plain(
                "If you can't catch the arrhythmia in the act, you can imitate it: "
                "pace from a spot and check whether the heartbeat it produces looks "
                "like the one you're trying to cure."
            ),
        ]
    )
    return


@app.cell
def _(bundle, np, plt, window_samples):
    # ILLUSTRATIVE. Normalised cross-correlation of ECG morphology — the method
    # behind PASO, on one lead instead of twelve. Not a PASO result.
    print(
        f"ECG channels available: {len(bundle.ecg_channel_names)} "
        f"({list(bundle.ecg_channel_names)}) — PASO needs 12"
    )

    def normalised_correlation(a, b):
        a = a - np.nanmean(a)
        b = b - np.nanmean(b)
        denominator = np.sqrt(np.nansum(a**2) * np.nansum(b**2))
        return np.nansum(a * b) / denominator if denominator > 0 else np.nan

    _segment = slice(*window_samples)
    _template = bundle.ecg[0, _segment]
    template_scores = np.array(
        [
            normalised_correlation(_template, bundle.ecg[i, _segment])
            for i in range(min(400, len(bundle.ecg)))
        ]
    )

    print(f"\ntemplate           : point {bundle.internal_names[0]}")
    print(f"beats compared     : {template_scores.size}")
    print(f"median correlation : {np.nanmedian(template_scores):.3f}")
    print(f"lowest correlation : {np.nanmin(template_scores):.3f}")
    print(f"above 0.95         : {np.nanmean(template_scores > 0.95) * 100:.1f}%")

    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.hist(
        template_scores[np.isfinite(template_scores)], bins=40,
        color="#8172b3", edgecolor="white", lw=0.4,
    )
    _ax.axvline(0.95, color="crimson", ls="--", label="0.95 match threshold")
    _ax.set_xlabel("normalised correlation with the template")
    _ax.set_ylabel("beats")
    _ax.set_title("Template correlation — the mathematics behind pace mapping (one lead)")
    _ax.legend()
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return (template_scores,)


@app.cell(hide_code=True)
def _(mo, np, takeaway, template_scores):
    mo.vstack(
        [
            mo.md(
                rf"""
    Look at how badly that performs as a discriminator: **{np.nanmean(template_scores > 0.95) * 100:.0f}%
    of beats clear the 0.95 threshold**, with a median correlation of
    {np.nanmedian(template_scores):.3f} and a *worst* case of
    {np.nanmin(template_scores):.3f}. If this were PASO, every site in the atrium
    would be declared a perfect match for the arrhythmia origin.

    Which is the point. These beats are all the same rhythm in the same patient, and
    projected onto one lead they are indistinguishable. A single ECG axis cannot
    separate sites whose activation patterns differ in directions that axis does not
    see. Twelve leads sample twelve different directions through the chest, and that
    redundancy is the entire source of PASO's specificity.

    The arithmetic above is correct. The conclusion you could draw from it would be
    worthless.
    """
            ),
            takeaway(
                "A validated method applied to insufficient data still produces "
                "confident numbers. Nothing in the output warns you; the warning has "
                "to come from knowing what the method needs."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What can and cannot be reconstructed from this file

    **Solid ground — measured directly**

    - Chamber geometry, to about a millimetre, *wherever the catheter went*.
    - Electrogram waveforms at the recorded sites, at 1 kHz.
    - Activation timing at those sites, subject to the annotation rule.
    - Signal amplitude, and through it a proxy for tissue health.
    - Lesion positions and their quality metrics.

    **Inference — derived, with real uncertainty**

    - Any value at a vertex that was not measured, which is most of them.
    - Conduction velocity: a gradient, of an interpolation, of annotated timings.
    - Scar classification, resting on thresholds and on direction-dependent bipolar
      voltage.
    - The arrhythmia mechanism, read from maps built on all of the above.

    **Out of reach from this data entirely**

    - Anything *transmural*. This is a surface map of a wall that has thickness, and
      circuits can run inside it, invisible here.
    - Anatomy the catheter never visited.
    - True 12-lead pace mapping, and the operator's own point tags.
    - Raw RF power, temperature and impedance — three channels that were never
      populated.
    - Durability. Whether these lesions still block conduction a year later needs
      follow-up, not a file.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 10 — Everything at once

    Anatomy, activation, substrate, conduction, and therapy: the same chamber under
    five different questions. This is roughly what an electrophysiologist is holding
    in their head simultaneously while deciding where to burn.
    """)
    return


@app.cell
def _(
    bundle,
    cv_speed,
    epkit,
    go,
    np,
    points,
    triangles,
    vertex_lat,
    vertex_voltage,
):
    # The full reconstruction, side by side.
    _spacing = 150.0
    _panels = [
        ("Anatomy", None, "Greys", None, None),
        (
            "Activation (ms)", vertex_lat, "Turbo_r",
            float(np.nanmin(vertex_lat)), float(np.nanmax(vertex_lat)),
        ),
        ("Voltage (mV)", np.clip(vertex_voltage, 0, 5), epkit.voltage_colorscale(), 0, 5),
        ("Conduction (mm/s)", np.clip(cv_speed, 0, 2000), "Viridis", 0, 2000),
    ]

    _traces = []
    for _offset, (_title, _field, _scale, _cmin, _cmax) in enumerate(_panels):
        _shifted = points.copy()
        _shifted[:, 0] += _offset * _spacing
        _traces.append(
            epkit.mesh_to_plotly(
                _shifted, triangles, scalars=_field, colorscale=_scale,
                cmin=_cmin, cmax=_cmax, name=_title, showscale=False,
            )
        )

    _traces.append(
        go.Scatter3d(
            x=bundle.tag_points[:, 0] + 3 * _spacing,
            y=bundle.tag_points[:, 1],
            z=bundle.tag_points[:, 2],
            mode="markers", marker=dict(size=5, color="red"), name="lesions",
        )
    )

    epkit.plotly_figure(
        _traces,
        title="Anatomy | Activation | Voltage | Conduction velocity + lesions",
        height=700,
    )
    return


@app.cell
def _(arrows, bundle, epkit, go, np, points, triangles, vertex_voltage):
    # One interactive view: substrate underneath, therapy on top, propagation arrows.
    epkit.plotly_figure(
        [
            epkit.mesh_to_plotly(
                points, triangles, scalars=np.clip(vertex_voltage, 0, 5),
                colorscale=epkit.voltage_colorscale(), cmin=0, cmax=5,
                colorbar_title="bipolar (mV)", opacity=0.95,
            ),
            go.Scatter3d(
                x=bundle.tag_points[:, 0],
                y=bundle.tag_points[:, 1],
                z=bundle.tag_points[:, 2],
                mode="markers",
                marker=dict(
                    size=6, color=np.asarray(bundle.tag_fti, dtype=float),
                    colorscale="Inferno", line=dict(width=0.5, color="white"),
                ),
                name="ablation lesions",
                text=[f"{_f:.0f} gs" for _f in bundle.tag_fti],
                hoverinfo="text",
            ),
            arrows,
        ],
        title="The complete picture: substrate, therapy, and propagation",
        height=720,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to keep

    Every clinical EP map is a small number of real measurements, interpolated
    across a surface, filtered through an annotation choice, and rendered in colours
    that imply far more certainty than the underlying data supports. **Knowing which
    parts are measured and which are inferred is the whole skill.**

    Five moments in this notebook made that concrete, and every one appeared only
    because we checked the data against what it claimed to be:

    | what was claimed | what the data said |
    |:---|:---|
    | bipolar = uniA − uniB | correlation 0.50, and not one point matching cleanly |
    | 1.5 mm point spacing means good coverage | 40% of the surface too far from any measurement to colour honestly |
    | `barDirection` is the electrode orientation | it is the surface normal — alignment 1.000, because it is the same array |
    | the export records RF power, temperature and impedance | three channels never populated, one of them a counter that would have plotted beautifully |
    | bipolar voltage depends strongly on wave direction | real, statistically significant, and a 7% effect |

    None of those were visible from the pretty picture. Each was one line of
    verification away.

    And the habit generalises far beyond electrophysiology: **when a field's name
    makes a promise, test the promise before you build on it.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go next

    - **[OpenEP](https://openep.io)** — the project these datasets come from. The
      MATLAB implementation is considerably more complete than the Python port used
      here.
    - **[openCARP](https://opencarp.org)** — simulate propagation rather than
      measure it, and ask what a circuit *would* do under an intervention.
    - **The gaps this notebook leaves open.** *Omnipolar* mapping attacks the
      bipolar direction-dependence of Part 6 directly, by reconstructing a
      direction-independent signal from an electrode array. *Charge-density* and
      *non-contact* mapping avoid the contact requirement altogether. *Ripple
      mapping* abandons annotation — the choice Part 2 showed moves the map — by
      animating raw voltage instead of reducing each point to one number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Appendix A — Glossary

    Everything defined along the way, in one place. Type to filter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    glossary = {
        "ablation": "Deliberately destroying a small patch of tissue to stop it conducting electricity. Here, by heating it with radiofrequency current.",
        "Ablation Index": "A proprietary score combining contact force, power and time, fitted to observed lesion depth. Watched in real time during burning.",
        "action potential": "The complete voltage excursion of one muscle cell: a fast flip from about −85 mV to +20 mV, a plateau, and a reset.",
        "activation map": "The chamber coloured by when each region was activated. Red early, blue late.",
        "annotation": "The chosen instant on an electrogram that counts as 'the wave arrived here'. A rule, not a measurement.",
        "arrhythmia": "Any abnormal heart rhythm.",
        "atrial fibrillation (AF)": "The commonest sustained arrhythmia: the atria quiver instead of beating, usually triggered from inside the pulmonary veins.",
        "atrium": "One of the two small upper chambers of the heart. This dataset maps the left one.",
        "bipolar": "A recording taken as the difference between two nearby electrodes. Cancels distant signal; sensitive to wave direction.",
        "breakthrough": "A wavefront surfacing onto the mapped surface from within the wall, rather than travelling across it.",
        "CARTO": "One of the commercial electroanatomic mapping systems. This dataset is a CARTO export.",
        "catheter": "A long steerable tube threaded through a vein into the heart, carrying electrodes at its tip.",
        "conduction velocity (CV)": "How fast the activation wave travels, in mm/s. Computed as the reciprocal of the timing gradient.",
        "contact force": "How hard the catheter tip presses on the wall, in grams. Measured by a sensor in modern tips.",
        "cycle length": "The interval between consecutive beats, in ms. 501 ms here, i.e. 120 per minute.",
        "depolarise": "For a cell, to become less negative inside — the switch flipping. For tissue, to be activated.",
        "durability": "Whether an ablation lesion is still blocking conduction months later. The central failure mode of AF ablation.",
        "electrogram": "A voltage trace recorded from inside the heart. An ECG is the same physics recorded from the skin.",
        "electrophysiology (EP)": "The study and treatment of the heart's electrical behaviour.",
        "far-field": "Signal arriving at an electrode from tissue that is not underneath it.",
        "fast anatomical mapping (FAM)": "Building the 3D shell by sweeping the catheter around the chamber and recording where it reached.",
        "fiducial": "A fixed reference instant — the zero of the time axis.",
        "focal": "An arrhythmia driven by one spot firing repeatedly. Cured by destroying that spot.",
        "force–time integral (FTI)": "Contact force multiplied by duration, in gram-seconds. A crude predictor of lesion depth.",
        "fractionation": "An electrogram with many small deflections instead of one sharp spike. The signature of slow, disorganised conduction through diseased tissue.",
        "free boundary": "An edge of the mesh belonging to only one triangle — the rim of a hole. In a real atrium, a vein or a valve.",
        "gap junction": "The protein channel that electrically connects one muscle cell to the next.",
        "interpolation": "Guessing values between measurements. Every coloured vertex that was not measured is interpolated.",
        "isochrone": "A contour of equal activation time. Crowded isochrones mean slow conduction.",
        "isthmus": "A narrow corridor of tissue that a reentrant circuit must pass through. The usual ablation target for reentry.",
        "local activation time (LAT)": "When the wave reached one point, measured against a reference beat.",
        "mitral valve": "The one-way door from the left atrium into the left ventricle. The largest hole in this mesh.",
        "myocyte": "A muscle cell.",
        "PASO": "CARTO's pace-mapping module: correlates a paced 12-lead ECG against the clinical arrhythmia's template.",
        "pace mapping": "Pacing from a candidate site to see whether it reproduces the arrhythmia's ECG pattern.",
        "pulmonary vein": "One of four vessels returning blood from the lungs into the left atrium. The usual source of AF triggers.",
        "pulmonary vein isolation (PVI)": "The standard AF operation: a ring of ablation around the veins so their triggers cannot escape into the atrium.",
        "radiofrequency (RF)": "The ~500 kHz alternating current used to heat and destroy tissue.",
        "reentry": "A wave that circulates indefinitely around a loop instead of dying out. Cured by cutting the loop.",
        "refractory": "Temporarily unable to be re-activated, just after activating.",
        "sentinel value": "A magic number standing in for missing data. Here, −10000. Silently ruins any statistic that does not mask it.",
        "substrate": "The tissue itself — healthy, diseased or scarred — as opposed to the rhythm running through it.",
        "transseptal": "Crossing from the right atrium to the left through the wall between them, by deliberate puncture.",
        "unipolar": "A recording of one electrode against a distant reference. Sees local and distant activity alike.",
        "voltage map": "The chamber coloured by signal amplitude, used as a proxy for how much healthy muscle is present.",
        "window of interest (WOI)": "The slice of a recording allowed to contain a valid activation.",
    }

    glossary_filter = mo.ui.text(
        placeholder="filter terms…", label="search", full_width=True
    )
    glossary_filter
    return glossary, glossary_filter


@app.cell(hide_code=True)
def _(glossary, glossary_filter, mo):
    _needle = glossary_filter.value.strip().lower()
    _matches = {
        term: meaning
        for term, meaning in glossary.items()
        if _needle in term.lower() or _needle in meaning.lower()
    }
    mo.md(
        f"*{len(_matches)} of {len(glossary)} terms*\n\n"
        "| term | meaning |\n|:---|:---|\n"
        + "\n".join(f"| **{term}** | {meaning} |" for term, meaning in _matches.items())
    )
    return


@app.cell(hide_code=True)
def _(bundle, mo):
    mo.md(
        rf"""
    ## Appendix B — Where this data comes from, and what was done to it

    The recordings are CARTO exports distributed with
    [OpenEP](https://openep.io) as MATLAB `.mat` files, about 67 MB for the two
    cases. Loading them needs `openep-py`, which needs `numpy<2`, MATLAB struct
    parsing and PyVista — none of which run in a browser.

    So the heavy lifting happens **once, offline**, in `scripts/build_bundle.py`,
    which writes a single compressed archive that this notebook reads:

    ```
    data/openep_dataset_1.mat   (51 MB, MATLAB)
        │  openep-py + scipy.io + pyvista        <- offline, numpy<2
        ▼
    public/ep_primer_data.npz   (12 MB, plain arrays)
        │  numpy + scipy only                    <- laptop or browser
        ▼
    this notebook
    ```

    **What the repacking does:** unwrap MATLAB structs into named arrays, store
    waveforms at reduced precision (bipolar and ECG as float32, the unipolar pair
    as float16, which halves the download and costs about three significant
    figures), and parse the `rfindex` lesion tags that `openep-py` does not read.

    **What it does not do:** no filtering, no resampling, no cleaning, no
    interpolation. Every derived quantity in this notebook is computed in the cells
    above, in front of you.

    **The three values computed at build time**, because they need `openep-py`
    itself and are used here as cross-checks, are labelled where they appear: the
    free boundaries of the mesh (Part 3), OpenEP's own activation field (Part 4),
    and its bipolar voltage field.

    Bundle in use: **{bundle.meta["source"]}**, {bundle.meta["n_points"]:,} points at
    {bundle.meta["sampling_frequency_hz"]:.0f} Hz, built against `openep-py`
    {bundle.meta["openep_version"]}.

    ### Running it yourself

    ```bash
    git clone https://github.com/vproHacks/ep && cd ep
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -r requirements.txt
    .venv/bin/marimo edit ep_primer.py
    ```

    That is enough to run everything, because `public/ep_primer_data.npz` is
    downloaded with the page. To rebuild the bundle from the original MATLAB files —
    which requires the pinned `numpy<2` stack — see `requirements-build.txt` and
    `scripts/build_bundle.py` in the repository.
    """
    )
    return


if __name__ == "__main__":
    app.run()
