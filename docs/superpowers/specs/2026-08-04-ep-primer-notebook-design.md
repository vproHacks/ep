# EP Primer Notebook — Design

**Date:** 2026-08-04
**Deliverable:** A self-contained educational Jupyter notebook teaching cardiac
electrophysiology data analysis from real clinical OpenEP data, culminating in
interactive 3D heart renders.

## Audience

An engineer comfortable with signals, linear algebra, and Python, but new to
cardiac biology. Physiology is built from first principles; mathematics is
stated directly without apology. Clinical significance is explained wherever a
number carries a clinical meaning (voltage thresholds, conduction velocity,
lesion durability).

## Goals

1. Explain what EP data physically *is* — from ion channel to sampled waveform.
2. Show what can be reconstructed from it, and at what confidence.
3. Render the reconstruction as an interactive 3D heart model inside the notebook.
4. Convey the clinical workflow an EP newcomer is expected to know.

## Non-goals

- Not a reusable library or CLI. It is a teaching document.
- No model fitting, no machine learning, no simulation.
- No modification of the OpenEP package itself.

## Data

Two real CARTO exports, bundled inside `openep-py` and re-saved by the OpenEP
authors with `triRep` cast to a struct (the `openep-examples` copies store
`triRep` as a MATLAB class object that OpenEP's own loader explicitly rejects —
do not use them).

| | Dataset 1 (**primary**) | Dataset 2 (secondary) |
|---|---|---|
| Vertices / triangles | 9,279 / 10,296 | 14,383 / 16,942 |
| Mapping points | 1,366 | 800 |
| Electrograms | 1,366 × 2500 @ 1 kHz | 800 × 2500 @ 1 kHz |
| Per-vertex NaN fraction | 41.5% | 38.9% |
| Rim vertices | 578 | 762 |
| LAT range | 24–121 ms | 20–175 ms |
| Bipolar voltage | 0.069–14.235 mV | 0.036–4.614 mV |
| `rfindex` lesion tags | **90, with full metrics** | absent |

Dataset 1 is primary because it carries the ablation lesion tags. Dataset 2
appears in the interpolation and comparison sections.

Both were collected with a 20-pole circular mapping catheter (electrode names
`20A_1-2` … `20A_19-20`).

### Data availability vs. requested topics

Honesty about provenance is a teaching goal, not a caveat. Every section states
whether a quantity is measured, derived, or illustrative.

| Topic | Availability | Treatment |
|---|---|---|
| Electrograms, geometry, LAT, voltage | Measured | Computed directly |
| Ablation lesion tags | Measured — 90 tags | Position, duration, avg force, max temp/power, impedance drop, FTI, lesion index |
| RF delivery time series | Measured — 17,258 samples | Power, impedance, temperature, force with catheter position/angles |
| FAM resolution | Derived | Mesh vertex spacing vs. electrical point spacing, quantified |
| Pattern acquisition | Derived | Inferred from catheter electrode names and point density |
| Complex point tags | **Field empty in both files** | Fractionation and double potentials detected from raw electrograms by us, then tagged |
| PASO / pace mapping | **Absent from OpenEP schema** | Explained in prose. The `ecg` array is **single-channel** (1366 × 2500 × 1, name `"ECG"`), not the 12 leads real PASO correlates — so the normalised cross-correlation mathematics is demonstrated on that one channel and labelled explicitly as illustrative of the method, not a reproduction of PASO |

No synthetic data is presented as clinical data anywhere.

## Environment

Everything is confined to the `ep/` folder.

```
ep/
├── .venv/                        uv-managed, Python 3.10
├── data/                         two .mat files (~67 MB), downloaded once and cached
├── docs/superpowers/specs/       this document
├── epkit.py                      rendering and I/O boilerplate only
├── requirements.txt              fully pinned
├── README.md                     setup and run instructions
└── ep_primer.ipynb               the deliverable
```

### Hard-won installation constraints

Discovered by testing, not assumed. All three are mandatory:

1. **`numpy<2`** — `openep/case/interpolators.py` references `np.NaN`, removed in
   NumPy 2.0. Import of the package fails outright otherwise.
2. **`setuptools<81`** — OpenEP's `setup.py` and `_datasets/openep_datasets.py`
   both import `pkg_resources`.
3. **`--no-build-isolation`** when installing OpenEP, so the build sees the
   pinned setuptools.

The dataset files are downloaded from `raw.githubusercontent.com` into `data/`
rather than relying on the installed package's resource paths, keeping the data
visible and the notebook self-explanatory.

Kernel is registered as `ep` via `ipykernel` so JupyterLab lists it by name.

## Implementation policy: minimal OpenEP surface, transparent mathematics

OpenEP is unmaintained (see Risks), and a teaching notebook that hides its
mathematics behind API calls teaches an API rather than the subject. So the
dependency is kept deliberately narrow:

- **OpenEP owns loading and the data model.** `load_openep_mat` and the `Case`
  object are the entry point — this is genuinely hard to replicate (the MATLAB
  struct traversal is fiddly) and is the least likely part to rot.
- **The analysis is ours**, written with numpy/scipy in visible notebook cells.
- **OpenEP serves as the reference check.** Wherever it offers an equivalent
  function, we compute ours first, then assert agreement against OpenEP's. That
  keeps the library valuable as a correctness oracle without letting it become a
  single point of failure.

The practical consequence: if `openep-py` breaks entirely, only Part 0 needs
rewriting — `epkit.load_case()` falls back to a direct `scipy.io` reader
(already required for `rfindex`), and every analytical cell keeps working.

**Available in OpenEP — used for loading, or as a cross-check against our own:**

- `openep.load_openep_mat`, `openep.to_pyvista` / `from_pyvista`, `case.create_mesh()`
- `case_routines`: `get_electrograms_at_points`, `get_mapping_points_within_woi`,
  `get_sample_indices_within_woi`, `calculate_voltage_from_electrograms`,
  `bipolar_from_unipolar_surface_points`, `interpolate_activation_time_onto_surface`,
  `interpolate_voltage_onto_surface`, `calculate_distance`,
  `calculate_points_within_distance`, `Interpolator`
- `mesh_routines`: `get_free_boundaries`, `calculate_field_area`,
  `calculate_mesh_volume`, `calculate_vertex_distance`, `calculate_vertex_path`,
  `low_field_area_per_region`, `mean_field_per_region`
- `draw_routines`: `draw_map`, `draw_free_boundaries`, `plot_electrograms`
- `case.fields` (per-vertex `bipolar_voltage`, `unipolar_voltage`,
  `local_activation_time`) and `case.ablation` (power, impedance, temperature, force)

**Hand-rolled, with the mathematics shown in the notebook:**

- **Scattered-data interpolation** onto the surface (inverse-distance / RBF with
  an explicit distance mask), cross-checked against
  `interpolate_activation_time_onto_surface`.
- **Voltage from electrograms** — peak-to-peak within the window of interest,
  cross-checked against `calculate_voltage_from_electrograms`.
- **Bipolar from unipolar** — the difference operator, shown empirically.
- **Conduction velocity.** The library has no CV function of any kind, so there
  is nothing to check against; correctness is argued from the mathematics and a
  synthetic planar-wave sanity test with known ground-truth velocity.
- **`rfindex` lesion tags.** `openep/io/readers.py` documents that `rfindex` is
  not loaded. Parsed with `scipy.io` into a small dataclass.
- **Re-annotation of LAT** by steepest negative deflection, for comparison
  against the system's stored annotation.
- **Complex-point detection** (fractionation index, double potentials).
- **Free-boundary tracing** from `isVertexAtRim` and mesh edge topology,
  cross-checked against `get_free_boundaries`.
- **Surface areas and distances**, cross-checked against `calculate_field_area`
  and `calculate_vertex_distance`.
- **Plotly mesh rendering** and the interpolation-method comparison.

## Notebook structure

Ten parts. Each opens with a plain-language statement of the question it answers
and closes with what was established.

**Part 0 — Orientation.** What a `Case` object contains, as one table.

**Part 1 — What EP data physically is.** Myocyte as a network of voltage-gated
conductances; the action potential; the heart as an excitable medium; why a
propagating wavefront produces an extracellular field a catheter can detect.
Then the mapping onto real arrays: 1,366 points × 2,500 samples at 1 kHz.

**Part 2 — The electrogram up close.** Unipolar vs. bipolar, with bipolar shown
empirically to be a spatial difference operator (a high-pass filter). Reference
channel, window of interest, and `LAT = mapAnnot − referenceAnnot`. Includes the
`-10000` sentinel for unannotated points — a real trap that silently corrupts a
map. We re-annotate by −dV/dt and quantify how far the map moves.

**Part 3 — Geometry.** Surface mesh fundamentals: manifoldness, normals,
adjacency, geodesic vs. Euclidean distance. FAM resolution quantified via
edge-length distribution against electrical point spacing. Free boundaries from
`isVertexAtRim` → pulmonary veins and mitral valve. **First 3D render.**

**Part 4 — Points to maps: interpolation.** The central reconstruction problem:
1,366 scattered samples → 9,279 vertices. Projection of points onto the surface,
with projection distance itself treated as a data-quality signal. Then the
honesty check: OpenEP's interpolator against linear and nearest-neighbour,
varying distance thresholds, and decimating 1,366 → 400 → 100 points to watch
the map degrade. Explains the 41.5% NaN fraction.

**Part 5 — Activation map.** LAT on the shell, isochrones, earliest and latest
sites, total activation time. Reading focal vs. macro-reentrant patterns and why
reentry shows early meeting late.

**Part 6 — Voltage and substrate.** Peak-to-peak recomputed from raw signal and
compared against stored values. Clinical thresholds (0.5 / 1.5 mV) for dense
scar, border zone, healthy tissue, with scar area computed via
`calculate_field_area`. Then bipolar's central weakness — dependence on wavefront
direction — demonstrated using the `barDirection` field.

**Part 7 — Conduction velocity.** Surface gradient of the LAT field,
CV = 1/|∇LAT|, derived and implemented per-triangle, then averaged to vertices.
Magnitude map plus vector arrows. Where slow-conduction corridors appear, and the
conditions under which CV estimates are untrustworthy (collision, breakthrough,
sparse sampling).

**Part 8 — Ablation and tagging.** RF delivery through a single lesion: force,
power, impedance, temperature time series. The 90 lesion tags rendered on the
shell, coloured by lesion quality. FTI and Ablation Index explained, with lesion
durability and pulmonary-vein reconnection as the clinical stakes. Then the
tagging taxonomy — ablation tags, point tags, anatomical landmarks — what OpenEP's
schema holds, the empty `electric.tags` field, and our derived complex-point
detection rendered on the map.

**Part 9 — Workflow context.** The lab sequence end to end: access → FAM geometry
→ mapping → diagnosis → pace mapping → ablation → validation remap. PASO
explained properly: why a 12-lead template correlation localises a focal source.
The normalised cross-correlation mathematics is then demonstrated on the single
available ECG channel, stated plainly as illustrating the method rather than
reproducing PASO — the data has one lead, not twelve. Point-by-point vs.
continuous acquisition and how each shows up in the data. Closes with an explicit
summary of what can and cannot be reconstructed.

**Part 10 — Finale.** Multi-layer interactive 3D: anatomy, LAT, voltage, CV, and
lesion tags composed into one view.

## Rendering

Every 3D figure is produced twice:

- **PyVista + trame** — live rotatable VTK view; the genuine OpenEP toolchain,
  via `draw_map` where applicable.
- **Plotly `Mesh3d`** — embedded in the saved `.ipynb`, so renders survive the
  kernel shutting down and the file being shared.

A shared `mesh_to_plotly()` helper in `epkit.py` keeps this plumbing out of the
teaching cells. If trame misbehaves under WSL2, an offscreen-PNG fallback path
keeps the notebook fully executable.

Colormaps follow EP convention: LAT uses a cyclic/rainbow scale with early as red;
voltage uses the clinical red-to-purple scar scale with thresholds at 0.5 and
1.5 mV.

## What lives in `epkit.py`

Boilerplate only. Teaching code stays visible in the notebook.

- `load_case(name)` — download-if-absent, cache, load via OpenEP
- `mesh_to_plotly(...)` — trimesh/`Mesh3d` conversion with colorbar handling
- `load_rfindex(path)` — the hand-rolled `rfindex` parser
- `show3d(...)` — dual PyVista/Plotly dispatch with PNG fallback
- MATLAB struct dereferencing helper

## Error handling

- Dataset download is checksum-verified by size and retried; failure raises with
  the manual download URL.
- Every cell must be runnable top-to-bottom on a clean kernel.
- NaN-heavy fields are masked explicitly and the masking is always shown, never
  silent.
- The `-10000` LAT sentinel is filtered at the single point of ingestion, with the
  filter demonstrated rather than hidden.

## Verification

- `Restart kernel & Run All` completes with no exceptions.
- Hand-rolled results cross-checked against OpenEP where both exist (voltage,
  interpolated fields).
- Reported numbers (vertex counts, NaN fractions, voltage ranges, lesion count)
  match the probe values recorded in this document.
- Both render paths produce output; Plotly figures persist in the saved file.

## Risks

| Risk | Mitigation |
|---|---|
| `openep-py` is unmaintained; further breakage | Pins are exact; core teaching maths is ours and independent of the library |
| trame/VTK unreliable headless under WSL2 | Plotly is the primary embedded path; PNG fallback |
| Notebook file size from embedded Plotly meshes | Decimate meshes for Plotly views; keep full resolution in PyVista |
| Length — ten parts is a lot | Each part standalone and independently runnable after Part 0 |
