# EP Primer — Accessibility Rewrite and marimo/WASM Port

**Date:** 2026-08-05
**Supersedes:** the delivery half of `2026-08-04-ep-primer-notebook-design.md`.
The data policy, the measured/derived/illustrative discipline and the
"cross-check against OpenEP, but write the mathematics out" stance are unchanged
and still binding.

## What changed, and why

Two complaints, one document.

1. **The notebook was inaccessible to anyone without cardiac training.** It
   opened at the level of "a cardiac myocyte holds roughly −85 mV" and never
   explained why a person would thread a wire into a heart in the first place.
   The audience in the original design — "an engineer comfortable with signals,
   new to cardiac biology" — was right; the execution assumed far more clinical
   context than that reader has.
2. **A 12 MB `.ipynb` in git, running on a pinned `numpy<2` + VTK stack, could
   not be shared.** Reading it required reproducing an installation whose
   constraints the README described as "hard-won".

## Audience, restated

A technically literate reader — comfortable reading numpy — with **no medical
background whatsoever**. Not a clinician, not a bioengineer. Someone who has
seen a heart diagram at school.

Nothing was removed to achieve this. The rigour, the failed textbook claims and
the uncomfortable numbers are all still there. Accessibility was added as
layers, not substituted for depth.

## Pedagogical devices

Each is used consistently, so the reader learns to recognise them:

| device | role |
|---|---|
| **Part 0 — Orientation** | New. The heart as an electrical machine, what an arrhythmia is, what the operation is for, what happens in the room, and a schematic of the chamber. Fifteen paragraphs that make the other ten parts legible. |
| **"In plain English" callouts** | A restatement of the preceding technical passage with no jargon, after every dense section. |
| **"Trap" callouts** | The specific way a reader would get this wrong. |
| **"Worth keeping" callouts** | The transferable lesson, usually generalising beyond electrophysiology. |
| **"Going deeper" accordions** | Optional depth — the engineering framing, the clinical rationale — collapsed so a beginner is not drowned. |
| **"Words used in this section" accordions** | Per-section jargon tables, plus a searchable glossary of ~45 terms in Appendix A. |
| **Analogies** | Fixed and reused: stadium wave (propagation), two microphones (bipolar subtraction), weather stations (interpolation), cave mapped in the dark (FAM), contour lines (conduction velocity). |
| **Live numbers** | Quantities narrated in prose are interpolated from the computation beside them wherever the number carries the argument, so the text cannot go stale against the data. |

## New analytical content

The accessibility work surfaced four claims that the old notebook asserted but
never computed. Each is now a cell:

- **A toy dipole-sheet model of an electrogram** (Part 1), with sliders for wave
  angle, electrode spacing and far-field contamination. It generates the classic
  biphasic unipolar deflection from first principles, shows the bipolar
  subtraction cancelling far field, and shows the bipolar signal vanishing at
  90° — which sets up Parts 6 and 7 physically rather than by assertion.
- **Cycle length by autocorrelation** (Part 5): 501 ms, constant to the
  millisecond, which answers the focal-vs-reentrant question numerically (97 ms
  of activation in a 501 ms cycle cannot be macro-reentry) and reveals that the
  map was collected under pacing.
- **Annotation disagreement against fractionation** (Part 8): the old text
  claimed annotation rules "disagree by tens of milliseconds" in diseased
  tissue. Measured: they agree within 5 ms on 98% of simple electrograms and
  disagree beyond 5 ms on 34% of fragmented ones.
- **A conduction-velocity self-test in the notebook** (Part 7), on a flat
  synthetic sheet, plus an accordion explaining why the same test on the curved
  atrial mesh does *not* return the planted answer — a real effect, not a bug.

Two claims were also corrected. The RF telemetry audit table now applies a
plausibility rule that agrees with its own narrative (the temperature channel
previously printed "usable" while the prose called it dead), and the
dataset-2 comparison no longer claims an identical pattern, because dataset 2's
temperature channel does reach 52 °C.

## Architecture

```
data/openep_dataset_1.mat        51 MB MATLAB, not in git
  │   scripts/build_bundle.py — openep-py, scipy.io, pyvista, numpy<2
  ▼
public/ep_primer_data.npz        12 MB plain arrays, committed
  │   epkit.load_bundle(path | url) — numpy only
  ▼
ep_primer.py                     marimo notebook
  │   marimo export html-wasm
  ▼
_site/                           GitHub Pages, runs in the reader's browser
```

**The bundle is the whole trick.** It draws a line between what needs the
awkward stack and what needs only numpy, and puts everything the notebook
touches on the numpy side. Consequences:

- The runtime stack loses `openep-py`, `pyvista`, `trame`, `vtk`, `jupyterlab`,
  `ipywidgets`, `nbformat`, `nbconvert` and the `numpy<2` pin.
- `scripts/build_notebook.py` is gone. A marimo notebook *is* a Python file, so
  the generator that existed to keep the notebook reviewable in git is
  redundant.
- PyVista/trame rendering is gone; Plotly is the only 3D engine, because it is
  the one that renders in a browser. The live-VTK view was never visible in a
  saved file anyway.
- Three values that genuinely need `openep-py` are computed at build time and
  stored so the notebook can still cross-check itself: the mesh free
  boundaries, OpenEP's activation field, and its bipolar voltage field. They are
  labelled as such where they are used.

**Precision.** Bipolar and ECG waveforms are stored float32 (exact enough that
peak-to-peak voltage still reproduces the system's stored value to 1e-8 mV, which
is asserted in `tests/test_bundle.py`); the unipolar pair is float16, halving the
download for arrays used only in correlations. Everything else is float32 or
int32. Total 12 MB, against 67 MB of MATLAB.

## marimo constraints worth knowing

- **Variables may be defined in only one cell.** Cell-local temporaries are
  underscore-prefixed, which marimo treats as private to the cell.
- **Cells form a DAG from static analysis**, not from file order, so the
  reading order and the execution order are independent. The declared function
  signatures are documentation; marimo recomputes references from the AST.
- **`mo.notebook_location()`** is a directory locally and a URL under Pyodide.
  `epkit.load_bundle` accepts either, which is why one line works in both.
- **Local packages are bundled automatically** by `marimo export html-wasm`,
  which builds a wheel of `epkit` into `public/wheels/`. The notebook's first
  cell keeps an explicit `micropip` fallback against a wheel the CI builds, so
  the published page does not depend on that convenience alone.
- **`tests/test_env.py` guards the browser contract**: importing `epkit` must
  not pull in `openep`, `pyvista` or `vtk`, and the PEP 723 header must not name
  them either. Those failures would otherwise appear only in a browser console,
  after a deploy.

## Verification

- `pytest tests/ -q` — 38 tests. The openep-dependent ones skip when the
  build stack or the `.mat` files are absent.
- `marimo export html --no-sandbox ep_primer.py -o /tmp/nb.html` executes every
  cell top to bottom; the export is scanned for tracebacks.
- `marimo export html-wasm ... && python -m http.server` — the page shell, the
  bundled wheel and the copied `public/` assets were verified in Chromium.
  Pyodide itself is fetched from a CDN that the build sandbox blocks, so the
  in-browser Python run is exercised by CI and on the published page, not
  locally.
