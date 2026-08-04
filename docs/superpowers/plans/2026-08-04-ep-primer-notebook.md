# EP Primer Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ep/ep_primer.ipynb`, a single educational notebook that teaches cardiac electrophysiology data analysis from real clinical OpenEP data and renders interactive 3D heart maps in-notebook.

**Architecture:** A thin support module (`epkit.py`) owns only data loading and rendering boilerplate, verified by pytest. All analytical mathematics lives in visible notebook cells, written with numpy/scipy, and is cross-checked against OpenEP where OpenEP offers an equivalent. OpenEP is confined to `load_openep_mat` plus the oracle role, so library rot cannot break the teaching content.

**Tech Stack:** Python 3.10, uv, numpy<2, scipy, pyvista+trame, plotly, matplotlib, openep-py (git), pytest, jupyterlab.

## Global Constraints

- Everything confined to `/home/vraj/Documents/personal/ep`. Write nothing outside it.
- **`numpy<2`** — mandatory. `openep/case/interpolators.py` uses `np.NaN`, removed in NumPy 2.0; `import openep` fails otherwise.
- **`setuptools<81`** — mandatory. OpenEP's `setup.py` and `_datasets/openep_datasets.py` import `pkg_resources`.
- **`--no-build-isolation`** — mandatory when installing openep, so the build sees the pinned setuptools.
- Primary dataset is **dataset 1**; dataset 2 is used only for comparison.
- Download `.mat` files from `raw.githubusercontent.com/openep/openep-py/main/openep/_datasets/OpenEP-MATLAB/` — **not** from `openep-examples`, whose copies store `triRep` as a MATLAB class object that OpenEP's loader rejects.
- Every notebook cell must run top-to-bottom on a clean kernel with no exceptions.
- Every quantity is labelled measured, derived, or illustrative. Never present derived or synthetic data as clinical measurement.
- Use they/them for any person referenced.

## Verified ground truth

These values were measured from the real files. Tests assert against them.

**Dataset 1 (primary):** 9,279 vertices; 10,296 triangles; 1,366 mapping points; egm `(1366, 2500)`; unipolar `(1366, 2500, 2)`; ecg `(1366, 2500, 1)` single channel named `"ECG"`; sampling 1000.0 Hz; WOI `[20, 180]`; reference annotation 2000; **22 of 1366 points carry the `-10000` LAT sentinel**; per-vertex fields 41.5% NaN; bipolar voltage 0.069–14.235 mV; LAT 24–121 ms; 578 rim vertices; **6 free boundaries** with lengths `[123.5, 70.5, 62.0, 61.6, 81.8, 50.8]` mm and areas `[1076.8, 360.7, 261.3, 277.0, 439.3, 186.3]` mm²; point→surface projection distance median 2.97 mm, p95 6.06 mm, max 6.98 mm; **90 `rfindex` lesion tags**; RF series 17,258 samples; force series 34,193 samples.

**Dataset 2:** 14,383 vertices; 16,942 triangles; 800 points; 38.9% NaN; 762 rim vertices; no `rfindex`.

**API notes:** point labels live in `electric.internal_names` (`P9`, `P10`, …); `electric.names` is all empty strings. `electric.surface.nearest_point` holds surface-projected coordinates; `electric.surface.normals` the normals. `electric.tags` is empty in both datasets.

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Exact pins |
| `README.md` | Setup and run instructions |
| `epkit/__init__.py` | Public re-exports |
| `epkit/data.py` | Download, cache, load cases; `rfindex` parser |
| `epkit/render.py` | Plotly/PyVista rendering helpers, EP colormaps |
| `tests/test_data.py` | Loader and `rfindex` parser tests |
| `tests/test_render.py` | Rendering helper tests |
| `tests/test_analysis.py` | Analytical helpers used by the notebook |
| `epkit/analysis.py` | The few analytical functions worth unit-testing (CV, fractionation) |
| `ep_primer.ipynb` | The deliverable |

`epkit` is a package rather than one module because rendering, data access, and analysis have genuinely different dependencies and change at different rates.

---

### Task 1: Environment and project skeleton

**Files:**
- Create: `requirements.txt`, `README.md`, `.gitignore`, `epkit/__init__.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: nothing
- Produces: a working `.venv` with openep importable; `epkit` package importable

- [ ] **Step 1: Initialise git and the virtual environment**

```bash
cd /home/vraj/Documents/personal/ep
git init
uv venv .venv --python 3.10
```

- [ ] **Step 2: Write `requirements.txt`**

```
# Pins are mandatory — see Global Constraints.
numpy<2
setuptools<81
wheel
scipy>=1.10
matplotlib>=3.7
plotly>=5.18
pyvista>=0.43
trame>=3.5
trame-vtk>=2.6
trame-vuetify>=2.4
ipywidgets>=8.1
jupyterlab>=4.0
ipykernel>=6.29
pytest>=8.0
openep @ git+https://github.com/openep/openep-py@main
```

- [ ] **Step 3: Install**

The two-phase install is required: setuptools must exist before openep's build runs.

```bash
cd /home/vraj/Documents/personal/ep
uv pip install --python .venv/bin/python "setuptools<81" wheel
uv pip install --python .venv/bin/python --no-build-isolation -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ep --display-name "EP Primer"
```

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
data/*.mat
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
```

- [ ] **Step 5: Write `epkit/__init__.py`**

```python
"""Support code for the EP primer notebook.

Only loading and rendering boilerplate lives here. The analytical
mathematics stays visible in the notebook cells.
"""

from epkit.data import DATA_DIR, LesionTags, load_case, load_rfindex

__all__ = ["DATA_DIR", "LesionTags", "load_case", "load_rfindex"]
```

- [ ] **Step 6: Write the failing environment test**

```python
# tests/test_env.py
import numpy as np


def test_numpy_is_pinned_below_2():
    assert np.__version__.startswith("1."), (
        f"openep requires numpy<2 (np.NaN); got {np.__version__}"
    )


def test_openep_imports():
    import openep

    assert hasattr(openep, "load_openep_mat")
```

- [ ] **Step 7: Run it**

Run: `cd /home/vraj/Documents/personal/ep && .venv/bin/python -m pytest tests/test_env.py -v`
Expected: PASS both. If `test_openep_imports` fails with `AttributeError: np.NaN`, the numpy pin did not take — reinstall.

- [ ] **Step 8: Write `README.md`**

````markdown
# EP Primer

An educational notebook on cardiac electrophysiology data analysis, built on
real clinical OpenEP datasets, with interactive 3D heart rendering.

## Setup

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python "setuptools<81" wheel
uv pip install --python .venv/bin/python --no-build-isolation -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ep --display-name "EP Primer"
```

The two-phase install is required: `openep-py` imports `pkg_resources` at build
time, and calls `np.NaN`, which NumPy 2 removed. Both pins are mandatory.

## Run

```bash
.venv/bin/jupyter lab ep_primer.ipynb
```

Select the "EP Primer" kernel. The first cell downloads ~67 MB of dataset on
first run and caches it in `data/`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```
````

- [ ] **Step 9: Commit**

```bash
cd /home/vraj/Documents/personal/ep
git add requirements.txt README.md .gitignore epkit/ tests/ docs/
git commit -m "chore: project skeleton, pinned environment, openep install"
```

---

### Task 2: Data loading and the rfindex parser

**Files:**
- Create: `epkit/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `DATA_DIR: pathlib.Path`
  - `load_case(which: int = 1) -> openep.data_structures.case.Case`
  - `LesionTags` dataclass with fields `points (N,3) float`, `duration (N,) float`, `avg_force (N,) float`, `max_temp (N,) float`, `max_power (N,) float`, `fti (N,) float`, `index_name: str`, `index_value (N,) float`
  - `load_rfindex(which: int = 1) -> LesionTags`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import numpy as np
import pytest

from epkit import load_case, load_rfindex


@pytest.fixture(scope="module")
def case1():
    return load_case(1)


def test_case1_geometry(case1):
    assert case1.points.shape == (9279, 3)
    assert case1.indices.shape == (10296, 3)


def test_case1_electrograms(case1):
    assert case1.electric.bipolar_egm.egm.shape == (1366, 2500)
    assert case1.electric.unipolar_egm.egm.shape == (1366, 2500, 2)
    assert case1.electric.frequency == 1000.0


def test_case1_ecg_is_single_channel(case1):
    """Real PASO needs 12 leads. This export has one. The notebook says so."""
    assert case1.electric.ecg.ecg.shape == (1366, 2500, 1)
    assert list(case1.electric.ecg.channel_names) == ["ECG"]


def test_case1_lat_sentinels_present(case1):
    ann = case1.electric.annotations
    lat = ann.local_activation_time - ann.reference_activation_time
    assert int(np.sum(lat <= -1000)) == 22, "the -10000 sentinel count is a teaching fact"


def test_case2_geometry():
    case2 = load_case(2)
    assert case2.points.shape == (14383, 3)


def test_load_rfindex_returns_90_lesions():
    tags = load_rfindex(1)
    assert tags.points.shape == (90, 3)
    for field in ("duration", "avg_force", "max_temp", "max_power", "fti"):
        assert getattr(tags, field).shape == (90,)


def test_rfindex_values_are_physiological():
    tags = load_rfindex(1)
    assert 0 < np.nanmedian(tags.avg_force) < 60, "grams-force"
    assert 20 < np.nanmedian(tags.max_power) < 60, "watts"
    assert 25 < np.nanmedian(tags.max_temp) < 60, "degrees C"


def test_rfindex_absent_for_dataset_2():
    with pytest.raises(ValueError, match="no rfindex"):
        load_rfindex(2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_case'`

- [ ] **Step 3: Implement `epkit/data.py`**

```python
"""Dataset download, caching, and loading.

The .mat files come from openep-py's bundled copies, in which the authors cast
triRep to a plain struct. The openep-examples copies store triRep as a MATLAB
class object that OpenEP's own loader explicitly refuses to read.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_BASE_URL = (
    "https://raw.githubusercontent.com/openep/openep-py/main/"
    "openep/_datasets/OpenEP-MATLAB"
)
_DATASETS = {
    1: ("openep_dataset_1.mat", 51_224_679),
    2: ("openep_dataset_2.mat", 16_371_668),
}


def _dataset_path(which: int) -> Path:
    if which not in _DATASETS:
        raise ValueError(f"dataset must be 1 or 2, got {which}")
    name, expected_size = _DATASETS[which]
    path = DATA_DIR / name

    if path.exists() and path.stat().st_size == expected_size:
        return path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE_URL}/{name}"
    print(f"Downloading {name} ({expected_size / 1e6:.0f} MB) ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {name}. Fetch it manually from {url} "
            f"and place it at {path}."
        ) from exc

    size = path.stat().st_size
    if size != expected_size:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name} downloaded with size {size}, expected {expected_size}. "
            f"Download it manually from {url}."
        )
    return path


def load_case(which: int = 1):
    """Load an OpenEP case, downloading and caching the dataset if needed."""
    import openep

    return openep.load_openep_mat(str(_dataset_path(which)))


def _deref(value):
    """Peel MATLAB struct wrapping until the payload is reached."""
    array = np.asarray(value)
    while array.dtype == object and array.size == 1 and array.dtype.names is None:
        array = np.asarray(array.ravel()[0])
    return array


@dataclass
class LesionTags:
    """Ablation lesion tags from the `rfindex` struct.

    openep-py does not load rfindex (documented in openep/io/readers.py), so
    this is parsed directly.
    """

    points: np.ndarray
    duration: np.ndarray
    avg_force: np.ndarray
    max_temp: np.ndarray
    max_power: np.ndarray
    fti: np.ndarray
    index_name: str
    index_value: np.ndarray

    def __len__(self) -> int:
        return len(self.points)


def load_rfindex(which: int = 1) -> LesionTags:
    """Parse the 90 ablation lesion tags out of the rfindex struct."""
    raw = scipy.io.loadmat(str(_dataset_path(which)))
    userdata = raw["userdata"]

    if "rfindex" not in (userdata.dtype.names or ()):
        raise ValueError(f"dataset {which} has no rfindex (no ablation tags)")

    rfindex = _deref(userdata.ravel()[0]["rfindex"])
    tag = _deref(rfindex.ravel()[0]["tag"]).ravel()[0]

    def field(name):
        return _deref(tag[name]).ravel().astype(float)

    index = _deref(tag["index"]).ravel()[0]
    index_name_raw = _deref(index["name"]).ravel()
    index_name = str(index_name_raw[0]) if index_name_raw.size else "index"

    return LesionTags(
        points=_deref(tag["X"]).astype(float),
        duration=field("time"),
        avg_force=field("avgForce"),
        max_temp=field("maxTemp"),
        max_power=field("maxPower"),
        fti=field("fti"),
        index_name=index_name,
        index_value=_deref(index["value"]).ravel().astype(float),
    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: PASS all 8. First run downloads ~67 MB.

If `test_rfindex_values_are_physiological` fails, print the actual medians and check the units before changing the bounds — the assertion encodes real physiology and a failure may mean the parser grabbed the wrong field.

- [ ] **Step 5: Commit**

```bash
git add epkit/data.py tests/test_data.py
git commit -m "feat: dataset download/cache, case loader, rfindex lesion-tag parser"
```

---

### Task 3: Analytical helpers worth unit-testing

**Files:**
- Create: `epkit/analysis.py`, `tests/test_analysis.py`
- Modify: `epkit/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `clean_lat(local, reference, sentinel_threshold=-1000.0) -> tuple[np.ndarray, np.ndarray]` returning `(lat, valid_mask)`
  - `surface_gradient(points, triangles, scalars) -> np.ndarray` — per-triangle gradient, shape `(n_tri, 3)`
  - `conduction_velocity(points, triangles, lat) -> tuple[np.ndarray, np.ndarray]` returning per-vertex `(speed_mm_per_s, direction (n_pts,3))`
  - `fractionation_index(egm, fs, window) -> np.ndarray` — deflection count per electrogram

Only these live in the module: they need real tests with known ground truth. Interpolation, voltage, and the rest stay in visible notebook cells.

- [ ] **Step 1: Write the failing tests**

The CV test uses a synthetic planar wave, the one case with an exact known answer. A wave travelling along +x at 1000 mm/s has LAT = x milliseconds when x is in mm.

```python
# tests/test_analysis.py
import numpy as np
import pytest

from epkit.analysis import (
    clean_lat,
    conduction_velocity,
    fractionation_index,
    surface_gradient,
)


def _flat_grid(n=21, spacing=1.0):
    """A flat triangulated sheet in the z=0 plane, n x n vertices."""
    xs = np.arange(n) * spacing
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris.append([a, b, d])
            tris.append([a, d, c])
    return points, np.array(tris)


def test_clean_lat_masks_sentinels():
    local = np.array([2073.0, 2045.0, -10000.0, 2100.0])
    reference = np.full(4, 2000.0)
    lat, valid = clean_lat(local, reference)
    assert valid.tolist() == [True, True, False, True]
    assert np.allclose(lat[valid], [73.0, 45.0, 100.0])
    assert np.isnan(lat[2])


def test_surface_gradient_of_linear_field_is_constant():
    points, tris = _flat_grid()
    scalars = points[:, 0]  # field increases 1 unit per mm in x
    grad = surface_gradient(points, tris, scalars)
    assert grad.shape == (len(tris), 3)
    assert np.allclose(grad, np.array([1.0, 0.0, 0.0]), atol=1e-8)


def test_conduction_velocity_recovers_planar_wave():
    """LAT = x ms on a wave along +x means 1 mm/ms = 1000 mm/s."""
    points, tris = _flat_grid()
    lat = points[:, 0]
    speed, direction = conduction_velocity(points, tris, lat)
    assert speed.shape == (len(points),)
    assert np.allclose(np.nanmedian(speed), 1000.0, rtol=1e-6)
    assert np.allclose(np.nanmedian(direction[:, 0]), 1.0, atol=1e-6)


def test_conduction_velocity_scales_with_wave_speed():
    """Halving the timing gradient doubles the velocity."""
    points, tris = _flat_grid()
    speed, _ = conduction_velocity(points, tris, 0.5 * points[:, 0])
    assert np.allclose(np.nanmedian(speed), 2000.0, rtol=1e-6)


def test_fractionation_index_counts_deflections():
    fs = 1000.0
    t = np.arange(0, 0.2, 1 / fs)
    window = (0, len(t))
    sharp = np.zeros_like(t)
    sharp[100] = 1.0
    fragmented = np.zeros_like(t)
    fragmented[[60, 80, 100, 120, 140]] = 0.5
    counts = fractionation_index(np.vstack([sharp, fragmented]), fs, window)
    assert counts[0] < counts[1], "fragmented signal must score higher"


def test_fractionation_index_shape():
    counts = fractionation_index(np.zeros((7, 500)), 1000.0, (0, 500))
    assert counts.shape == (7,)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'epkit.analysis'`

- [ ] **Step 3: Implement `epkit/analysis.py`**

```python
"""Analytical routines that carry their own tests.

OpenEP has no conduction-velocity function of any kind, so the CV code here is
validated against a synthetic planar wave with a known answer rather than
against the library.
"""

from __future__ import annotations

import numpy as np


def clean_lat(local, reference, sentinel_threshold: float = -1000.0):
    """Local activation time relative to the reference, with sentinels masked.

    Unannotated points carry a -10000 sentinel. Left in place it silently drags
    a whole activation map. Returns (lat, valid_mask) with invalid entries NaN.
    """
    local = np.asarray(local, dtype=float)
    reference = np.asarray(reference, dtype=float)
    lat = local - reference
    valid = (lat > sentinel_threshold) & np.isfinite(lat)
    lat = np.where(valid, lat, np.nan)
    return lat, valid


def surface_gradient(points, triangles, scalars):
    """Per-triangle gradient of a scalar field on a triangulated surface.

    For a linear field over a triangle, the in-plane gradient is obtained by
    solving the two edge equations
        (p1 - p0) . g = s1 - s0
        (p2 - p0) . g = s2 - s0
    together with n . g = 0, which pins g to the triangle plane.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)
    scalars = np.asarray(scalars, dtype=float)

    p0, p1, p2 = (points[triangles[:, i]] for i in range(3))
    s0, s1, s2 = (scalars[triangles[:, i]] for i in range(3))

    e1, e2 = p1 - p0, p2 - p0
    normal = np.cross(e1, e2)
    norm_len = np.linalg.norm(normal, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normal = normal / norm_len

    matrix = np.stack([e1, e2, normal], axis=1)          # (n_tri, 3, 3)
    rhs = np.stack([s1 - s0, s2 - s0, np.zeros(len(triangles))], axis=1)

    grad = np.full((len(triangles), 3), np.nan)
    ok = np.isfinite(rhs).all(axis=1) & np.isfinite(matrix).all(axis=(1, 2))
    ok &= np.abs(np.linalg.det(matrix)) > 1e-12
    if np.any(ok):
        grad[ok] = np.linalg.solve(matrix[ok], rhs[ok])
    return grad


def conduction_velocity(points, triangles, lat):
    """Per-vertex conduction speed (mm/s) and unit propagation direction.

    Speed is 1/|grad LAT| with LAT in ms and coordinates in mm, scaled to mm/s.
    Propagation runs down the timing gradient, so direction is +grad normalised:
    the gradient points toward later activation, which is where the wave goes.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)

    grad = surface_gradient(points, triangles, lat)
    magnitude = np.linalg.norm(grad, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tri_speed = np.where(magnitude > 0, 1000.0 / magnitude, np.nan)
        tri_dir = grad / magnitude[:, None]

    speed = _average_to_vertices(tri_speed[:, None], triangles, len(points))[:, 0]
    direction = _average_to_vertices(tri_dir, triangles, len(points))

    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.where(lengths > 0, direction / lengths, np.nan)
    return speed, direction


def _average_to_vertices(tri_values, triangles, n_points):
    """Mean of incident triangle values at each vertex, ignoring NaN."""
    n_cols = tri_values.shape[1]
    totals = np.zeros((n_points, n_cols))
    counts = np.zeros((n_points, n_cols))
    valid = np.isfinite(tri_values)
    filled = np.where(valid, tri_values, 0.0)

    for corner in range(3):
        idx = triangles[:, corner]
        np.add.at(totals, idx, filled)
        np.add.at(counts, idx, valid.astype(float))

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)


def fractionation_index(egm, fs, window):
    """Count significant deflections in each electrogram within `window`.

    A fractionated signal - the electrical signature of slow, disorganised
    conduction through diseased tissue - shows many deflections where healthy
    tissue shows one sharp biphasic spike.
    """
    egm = np.atleast_2d(np.asarray(egm, dtype=float))
    start, stop = window
    segment = egm[:, start:stop]

    derivative = np.gradient(segment, 1.0 / fs, axis=1)
    peak = np.nanmax(np.abs(derivative), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(peak > 0, np.abs(derivative) / peak, 0.0)

    above = normalised > 0.2
    crossings = np.diff(above.astype(int), axis=1) == 1
    return crossings.sum(axis=1).astype(float)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: PASS all 6.

- [ ] **Step 5: Re-export from `epkit/__init__.py`**

```python
"""Support code for the EP primer notebook.

Only loading, rendering, and tested analytical helpers live here. The
explanatory mathematics stays visible in the notebook cells.
"""

from epkit.analysis import (
    clean_lat,
    conduction_velocity,
    fractionation_index,
    surface_gradient,
)
from epkit.data import DATA_DIR, LesionTags, load_case, load_rfindex

__all__ = [
    "DATA_DIR",
    "LesionTags",
    "clean_lat",
    "conduction_velocity",
    "fractionation_index",
    "load_case",
    "load_rfindex",
    "surface_gradient",
]
```

- [ ] **Step 6: Commit**

```bash
git add epkit/analysis.py epkit/__init__.py tests/test_analysis.py
git commit -m "feat: CV, LAT cleaning, and fractionation helpers with ground-truth tests"
```

---

### Task 4: Rendering helpers

**Files:**
- Create: `epkit/render.py`, `tests/test_render.py`
- Modify: `epkit/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LAT_CMAP`, `VOLTAGE_CMAP`: matplotlib colormap names/objects
  - `mesh_to_plotly(points, triangles, scalars=None, *, colorscale="Turbo", cmin=None, cmax=None, name="", opacity=1.0, colorbar_title="") -> plotly.graph_objects.Mesh3d`
  - `plotly_figure(traces, title="", height=650) -> plotly.graph_objects.Figure` — equal-aspect 3D scene, dark/light neutral
  - `voltage_colorscale() -> list` — clinical scar scale with 0.5/1.5 mV breakpoints
  - `decimate(points, triangles, scalars, target_faces) -> tuple` — for keeping embedded Plotly small
  - `show_pyvista(mesh, scalars=None, **kwargs)` — trame view with offscreen-PNG fallback

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_render.py
import numpy as np
import plotly.graph_objects as go
import pytest

from epkit.render import decimate, mesh_to_plotly, plotly_figure, voltage_colorscale


def _tetra():
    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return points, tris


def test_mesh_to_plotly_returns_mesh3d_with_correct_topology():
    points, tris = _tetra()
    mesh = mesh_to_plotly(points, tris)
    assert isinstance(mesh, go.Mesh3d)
    assert len(mesh.x) == 4
    assert len(mesh.i) == 4


def test_mesh_to_plotly_attaches_scalars():
    points, tris = _tetra()
    mesh = mesh_to_plotly(points, tris, scalars=np.array([0.0, 1, 2, 3]))
    assert mesh.intensity is not None
    assert len(mesh.intensity) == 4


def test_mesh_to_plotly_handles_nan_scalars():
    """41.5% of real vertices are NaN. This must not raise."""
    points, tris = _tetra()
    mesh = mesh_to_plotly(points, tris, scalars=np.array([np.nan, 1.0, np.nan, 3.0]))
    assert isinstance(mesh, go.Mesh3d)


def test_plotly_figure_has_equal_aspect():
    points, tris = _tetra()
    fig = plotly_figure([mesh_to_plotly(points, tris)])
    assert fig.layout.scene.aspectmode == "data"


def test_voltage_colorscale_is_normalised():
    scale = voltage_colorscale()
    positions = [stop[0] for stop in scale]
    assert positions[0] == 0.0
    assert positions[-1] == 1.0
    assert positions == sorted(positions)


def test_decimate_reduces_face_count():
    rng = np.random.default_rng(0)
    n = 400
    points = rng.random((n, 3))
    tris = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)])
    scalars = rng.random(n)
    dp, dt, ds = decimate(points, tris, scalars, target_faces=50)
    assert len(dt) <= len(tris)
    assert len(ds) == len(dp)


def test_decimate_is_a_noop_when_already_small():
    points, tris = _tetra()
    scalars = np.arange(4.0)
    dp, dt, ds = decimate(points, tris, scalars, target_faces=1000)
    assert len(dt) == len(tris)
    assert len(dp) == len(points)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'epkit.render'`

- [ ] **Step 3: Implement `epkit/render.py`**

```python
"""Rendering boilerplate: Plotly for embedded views, PyVista for live ones.

Every 3D figure in the notebook is produced twice. PyVista/trame gives a live
rotatable VTK view; Plotly Mesh3d survives the kernel shutting down, so the
saved notebook still shows its renders.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Early activation red through to late blue, the convention on clinical systems.
LAT_CMAP = "Turbo_r"
VOLTAGE_CMAP = "jet_r"

_BACKGROUND = "rgba(0,0,0,0)"


def voltage_colorscale(low=0.5, high=1.5, vmax=5.0):
    """Clinical bipolar-voltage scale.

    Below `low` mV is dense scar (red), above `high` mV is healthy myocardium
    (purple/blue), between them is border zone. Breakpoints are placed at the
    real thresholds rather than spread evenly.
    """
    lo, hi = low / vmax, high / vmax
    return [
        [0.0, "rgb(190,30,30)"],
        [max(lo - 1e-6, 0.0), "rgb(230,90,50)"],
        [lo, "rgb(240,200,60)"],
        [(lo + hi) / 2, "rgb(70,190,90)"],
        [hi, "rgb(60,140,220)"],
        [1.0, "rgb(120,60,180)"],
    ]


def mesh_to_plotly(
    points,
    triangles,
    scalars=None,
    *,
    colorscale="Turbo",
    cmin=None,
    cmax=None,
    name="",
    opacity=1.0,
    colorbar_title="",
    showscale=True,
):
    """Wrap a triangulated surface as a Plotly Mesh3d trace."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)

    kwargs = dict(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name=name,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
    )

    if scalars is None:
        kwargs.update(color="rgb(200,180,175)")
    else:
        scalars = np.asarray(scalars, dtype=float)
        finite = scalars[np.isfinite(scalars)]
        kwargs.update(
            intensity=scalars,
            colorscale=colorscale,
            cmin=cmin if cmin is not None else (finite.min() if finite.size else 0.0),
            cmax=cmax if cmax is not None else (finite.max() if finite.size else 1.0),
            showscale=showscale,
            colorbar=dict(title=colorbar_title, thickness=14, len=0.7),
        )
    return go.Mesh3d(**kwargs)


def plotly_figure(traces, title="", height=650):
    """Assemble traces into an equal-aspect 3D figure."""
    fig = go.Figure(data=list(traces))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        paper_bgcolor=_BACKGROUND,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=_BACKGROUND,
        ),
    )
    return fig


def decimate(points, triangles, scalars, target_faces=20000):
    """Reduce face count for embedded Plotly views.

    Full resolution stays in the PyVista path; this only keeps the saved
    notebook from bloating.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)
    scalars = None if scalars is None else np.asarray(scalars, dtype=float)

    if len(triangles) <= target_faces:
        return points, triangles, scalars

    import pyvista as pv

    faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
    mesh = pv.PolyData(points, faces)
    if scalars is not None:
        mesh.point_data["scalars"] = scalars

    reduction = 1.0 - (target_faces / len(triangles))
    reduced = mesh.decimate(reduction).triangulate().clean()

    new_points = np.asarray(reduced.points)
    new_tris = np.asarray(reduced.faces).reshape(-1, 4)[:, 1:]
    new_scalars = (
        np.asarray(reduced.point_data["scalars"])
        if scalars is not None and "scalars" in reduced.point_data
        else None
    )
    return new_points, new_tris, new_scalars


def show_pyvista(mesh, scalars=None, *, cmap="turbo", title="", clim=None, **kwargs):
    """Live PyVista/trame view, falling back to a static image if that fails.

    Headless WSL2 sometimes cannot start a trame view; the fallback keeps the
    notebook runnable end to end.
    """
    import pyvista as pv

    try:
        plotter = pv.Plotter(notebook=True, window_size=(760, 560))
        plotter.add_mesh(
            mesh, scalars=scalars, cmap=cmap, clim=clim, smooth_shading=True, **kwargs
        )
        if title:
            plotter.add_text(title, font_size=10)
        plotter.set_background("white")
        return plotter.show(jupyter_backend="trame", return_viewer=True)
    except Exception as exc:  # noqa: BLE001 - fall back rather than break the notebook
        print(f"[trame unavailable: {exc}] falling back to a static image")
        plotter = pv.Plotter(off_screen=True, window_size=(760, 560))
        plotter.add_mesh(
            mesh, scalars=scalars, cmap=cmap, clim=clim, smooth_shading=True, **kwargs
        )
        plotter.set_background("white")
        return plotter.show(screenshot=True, return_img=True)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_render.py -v`
Expected: PASS all 8.

- [ ] **Step 5: Add rendering exports to `epkit/__init__.py`**

Append to the imports and `__all__`:

```python
from epkit.render import (
    LAT_CMAP,
    VOLTAGE_CMAP,
    decimate,
    mesh_to_plotly,
    plotly_figure,
    show_pyvista,
    voltage_colorscale,
)
```

Add `"LAT_CMAP"`, `"VOLTAGE_CMAP"`, `"decimate"`, `"mesh_to_plotly"`, `"plotly_figure"`, `"show_pyvista"`, `"voltage_colorscale"` to `__all__`.

- [ ] **Step 6: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest tests/ -v
git add epkit/render.py epkit/__init__.py tests/test_render.py
git commit -m "feat: Plotly/PyVista rendering helpers with clinical colour scales"
```

---

### Task 5: Notebook Parts 0-2 — orientation, physiology, electrograms

**Files:**
- Create: `ep_primer.ipynb`

Build the notebook with `nbformat` in a script, then execute it. Written as `scripts/build_notebook.py` so the notebook is reproducible rather than hand-assembled; the script appends cells part by part across Tasks 5-9.

**Interfaces:**
- Consumes: `epkit.load_case`, `epkit.clean_lat`
- Produces: `ep_primer.ipynb` containing Parts 0-2; variables `case`, `egm`, `lat`, `valid` available to later parts

- [ ] **Step 1: Create the notebook build script skeleton**

```python
# scripts/build_notebook.py
"""Assemble ep_primer.ipynb.

Keeping the notebook generated from source makes it reviewable in git and
reproducible. Run: .venv/bin/python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

NOTEBOOK = Path(__file__).resolve().parent.parent / "ep_primer.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def write() -> None:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "EP Primer",
        "language": "python",
        "name": "ep",
    }
    NOTEBOOK.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({len(cells)} cells)")
```

- [ ] **Step 2: Add Part 0 — title and orientation**

Append to `scripts/build_notebook.py`:

```python
md(r"""
# Electrophysiology Data, End to End

**What this notebook is.** A working introduction to cardiac electrophysiology
(EP) data: what it physically is, how it is turned into the coloured 3D heart
maps you see in an EP lab, and — just as important — how much of those maps is
measurement and how much is inference.

Everything here runs on **real clinical data**: two CARTO exports distributed
with the [OpenEP](https://openep.io) project, from catheter ablation procedures.

**How to read it.** Every quantity is labelled:

- **Measured** — recorded by the mapping system.
- **Derived** — computed here from measured data.
- **Illustrative** — a demonstration of a method the data cannot fully support.

**A note on the library.** We use `openep-py` to load the data and as a
correctness check, but the analysis is written out explicitly with numpy and
scipy. The point is to see the mathematics, not to call an API.
""")

code("""
import warnings

import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from scipy import signal, spatial

import epkit
from epkit import clean_lat, load_case, mesh_to_plotly, plotly_figure

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)

case = load_case(1)          # primary dataset: has the ablation lesion tags
case2 = load_case(2)         # secondary, used for comparison
print(case)
""")

md(r"""
## Part 0 — What is actually in the file

An EP study export is three things bolted together: a **geometry** (the shell),
a set of **electrical recordings** (the points), and a record of **therapy
delivered** (the ablation). Everything else is derived from those.
""")

code("""
electric = case.electric

summary = [
    ("Geometry: vertices",        case.points.shape[0]),
    ("Geometry: triangles",       case.indices.shape[0]),
    ("Mapping points",            electric.n_points),
    ("Samples per electrogram",   electric.n_samples),
    ("Sampling frequency (Hz)",   electric.frequency),
    ("Recording duration (s)",    electric.n_samples / electric.frequency),
    ("Bipolar electrogram array", electric.bipolar_egm.egm.shape),
    ("Unipolar electrogram array", electric.unipolar_egm.egm.shape),
    ("ECG array",                 electric.ecg.ecg.shape),
    ("Ablation samples",          len(case.ablation.times)),
]
width = max(len(label) for label, _ in summary)
for label, value in summary:
    print(f"{label:<{width}} : {value}")
""")

md(r"""
Note the shape of the problem. There are **1,366 electrical recordings** but
**9,279 geometry vertices**. Every coloured map you will ever see in an EP lab is
an interpolation from the former onto the latter — roughly seven vertices coloured
for every one actually measured. Part 4 takes that seriously.
""")
```

- [ ] **Step 3: Add Part 1 — the physics, from ion channel to electrode**

```python
md(r"""
## Part 1 — What EP data physically is

### The cell

A cardiac myocyte holds roughly **-85 mV** across its membrane at rest, set by
ion pumps maintaining concentration gradients. The membrane is studded with
voltage-gated ion channels — conductances that open and close as a function of
the very voltage they control.

That feedback is the whole story. Depolarise the membrane past about -65 mV and
sodium channels open, admitting Na⁺, depolarising it further, opening more
channels. The result is a near-instantaneous upstroke to about +20 mV: the
**action potential**. Calcium channels sustain a plateau (this is what makes
cardiac action potentials last ~200-300 ms rather than the ~1 ms of a neuron),
then potassium channels repolarise the cell.

For an engineer: it is a relaxation oscillator with a refractory period, and the
refractory period is what stops the heart from simply oscillating uncontrollably.

### The tissue

Myocytes are electrically coupled through gap junctions, so a cell reaching
threshold drags its neighbours with it. The tissue is an **excitable medium**: a
depolarisation wavefront propagates outward at roughly **0.3-1.0 m/s**, and
behind it the refractory tail leaves tissue briefly unexcitable.

Arrhythmia is what happens when that wave misbehaves. If a wavefront finds a path
back to tissue that has recovered, it re-excites it, and you have **reentry** — a
self-sustaining circuit. Diseased tissue conducts slowly, which makes reentry
easier, because slow conduction gives tissue ahead of the wave time to recover.
This is why the slow-conduction regions we map in Part 7 matter clinically.

### The measurement

A moving wavefront is a moving boundary between depolarised and resting tissue —
a travelling charge separation, which produces a **potential field in the blood
and tissue around it**. An electrode in the chamber samples that field.

This is the key point: the catheter does **not** measure the action potential of
any cell. It measures the extracellular field produced by many cells depolarising
together. What you get is roughly proportional to the *rate of change* of nearby
transmembrane potential, weighted by 1/distance. So:

- Fast, synchronous depolarisation nearby → a large sharp deflection.
- Slow or disorganised depolarisation → a small, smeared, fragmented signal.
- Dead scar tissue → almost nothing, just far-field signal from elsewhere.

That relationship is what makes the whole enterprise work: **signal amplitude
becomes a proxy for tissue health, and deflection timing becomes a proxy for when
the wave arrived.**
""")

code("""
# MEASURED. One electrogram, and what the sampling actually looks like.
fs = electric.frequency
times_ms = np.arange(electric.n_samples) / fs * 1000.0

point = 500
bipolar = electric.bipolar_egm.egm[point]

print(f"Point label      : {electric.internal_names[point]}")
print(f"Sampling         : {fs:.0f} Hz -> {1000 / fs:.1f} ms per sample")
print(f"Duration         : {times_ms[-1] / 1000:.2f} s")
print(f"Amplitude range  : {bipolar.min():.3f} to {bipolar.max():.3f} mV")
""")

code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].plot(times_ms, electric.unipolar_egm.egm[point, :, 0], lw=0.8, color="#1f77b4")
axes[0].set_ylabel("Unipolar A\\n(mV)")
axes[0].set_title(f"Point {electric.internal_names[point]} — the same event, three ways")

axes[1].plot(times_ms, electric.unipolar_egm.egm[point, :, 1], lw=0.8, color="#ff7f0e")
axes[1].set_ylabel("Unipolar B\\n(mV)")

axes[2].plot(times_ms, bipolar, lw=0.8, color="#2ca02c")
axes[2].set_ylabel("Bipolar\\n(mV)")
axes[2].set_xlabel("Time (ms)")

for ax in axes:
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
""")
```

- [ ] **Step 4: Add Part 2 — electrograms, bipolar derivation, annotation, sentinels**

```python
md(r"""
## Part 2 — The electrogram up close

### Unipolar vs bipolar: a spatial high-pass filter

A **unipolar** recording measures one electrode against a distant reference. It
sees local activity, but also everything else in the heart — *far-field* signal.

A **bipolar** recording is the difference between two closely spaced electrodes.
Distant sources affect both electrodes almost equally, so subtracting cancels
them; local sources affect them differently, so local signal survives.

In signal-processing terms, subtracting two spatially offset samples of a field
is a **discrete spatial derivative** — a high-pass filter in space. It buys
locality at a price we will pay in Part 6: the output depends on the *direction*
the wavefront crosses the electrode pair.

Let us confirm the derivation is exactly that difference.
""")

code("""
# DERIVED. Bipolar should be the difference of the two unipolar channels.
uni_a = electric.unipolar_egm.egm[:, :, 0]
uni_b = electric.unipolar_egm.egm[:, :, 1]
derived_bipolar = uni_a - uni_b
stored_bipolar = electric.bipolar_egm.egm

residual = np.abs(derived_bipolar - stored_bipolar)
print(f"max |derived - stored| : {np.nanmax(residual):.6f} mV")
print(f"median |residual|      : {np.nanmedian(residual):.6f} mV")
print(f"correlation            : {np.corrcoef(derived_bipolar.ravel(), stored_bipolar.ravel())[0, 1]:.6f}")
""")

md(r"""
### Timing: the reference, the window, and the annotation

A single electrogram tells you a wave arrived, not *when* in the cardiac cycle.
Timing is only meaningful relative to a fixed fiducial, so every point is timed
against a **reference channel** — typically a stable intracardiac electrode.

Three numbers per point:

- `referenceAnnot` — the fiducial on the reference channel (2000 samples here).
- `mapAnnot` — the moment of local activation at this point.
- `woi` — the **window of interest**, the slice of the recording that is allowed
  to contain valid activation.

**Local activation time** is the difference. And here is a trap that will silently
ruin a map.
""")

code("""
# MEASURED, then cleaned. Unannotated points carry a -10000 sentinel.
annotations = electric.annotations
raw_lat = annotations.local_activation_time - annotations.reference_activation_time

print(f"raw LAT range     : {raw_lat.min():.0f} to {raw_lat.max():.0f} ms")
print(f"sentinel points   : {int(np.sum(raw_lat <= -1000))} of {raw_lat.size}")

lat, valid = clean_lat(
    annotations.local_activation_time, annotations.reference_activation_time
)
print(f"cleaned LAT range : {np.nanmin(lat):.0f} to {np.nanmax(lat):.0f} ms")
print()
print(f"mean WITH sentinels    : {raw_lat.mean():>10.1f} ms   <- meaningless")
print(f"mean WITHOUT sentinels : {np.nanmean(lat):>10.1f} ms   <- correct")
""")

md(r"""
Twenty-two points out of 1,366 — 1.6% of the data — drag the naive mean to a
physically impossible value. Any statistic, interpolation, or colour scale
computed without masking them is wrong, and wrong in a way that still produces a
plausible-looking picture. **This is the single most common way to silently
corrupt an activation map.**

### Re-annotating it ourselves

The mapping system chose `mapAnnot` for us. On a bipolar electrogram, activation
is conventionally annotated at the **peak absolute amplitude** or the steepest
deflection. Let us apply that rule ourselves and see how far we land from the
system.
""")

code("""
# DERIVED. Re-annotate by steepest deflection within the window of interest.
woi = annotations.window_of_interest
reference = annotations.reference_activation_time

our_lat = np.full(electric.n_points, np.nan)
for i in range(electric.n_points):
    start = int(reference[i] + woi[i, 0])
    stop = int(reference[i] + woi[i, 1])
    start, stop = max(start, 0), min(stop, electric.n_samples)
    if stop - start < 3:
        continue
    segment = stored_bipolar[i, start:stop]
    our_lat[i] = start + int(np.argmax(np.abs(np.gradient(segment)))) - reference[i]

both = valid & np.isfinite(our_lat)
difference = our_lat[both] - lat[both]
print(f"points compared      : {both.sum()}")
print(f"median difference    : {np.median(difference):>7.1f} ms")
print(f"IQR                  : {np.percentile(difference, 25):.1f} to {np.percentile(difference, 75):.1f} ms")
print(f"within +/-5 ms       : {np.mean(np.abs(difference) <= 5) * 100:.1f}%")
""")

md(r"""
The annotation rule is a **choice**, and the choice moves the map. Where the
electrogram is a single sharp spike, every rule agrees. Where it is fragmented —
exactly the diseased tissue we most want to characterise — rules disagree by tens
of milliseconds. That uncertainty propagates into every downstream map.
""")

write()
```

- [ ] **Step 5: Build and execute the notebook so far**

```bash
cd /home/vraj/Documents/personal/ep
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ep --ExecutePreprocessor.timeout=600 ep_primer.ipynb
```

Expected: completes with no exceptions. Confirm the printed residual between derived and stored bipolar is near zero, and that the sentinel count prints 22.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_notebook.py ep_primer.ipynb
git commit -m "feat(notebook): Parts 0-2 — orientation, physiology, electrograms"
```

---

### Task 6: Notebook Part 3 — geometry, FAM resolution, first 3D render

**Files:**
- Modify: `scripts/build_notebook.py`, `ep_primer.ipynb`

**Interfaces:**
- Consumes: `case`, `electric` from Part 0; `epkit.mesh_to_plotly`, `plotly_figure`, `show_pyvista`
- Produces: `mesh` (pyvista PolyData), `free_boundaries` available to later parts

- [ ] **Step 1: Add the geometry narrative and mesh statistics**

Insert before `write()` in `scripts/build_notebook.py`:

```python
md(r"""
## Part 3 — The shell: where the geometry comes from

The coloured heart you see on an EP screen is a **triangulated surface mesh**
built during the procedure, not a scan. The operator sweeps the catheter around
the inside of the chamber and the system accumulates the outermost positions the
catheter reached — **fast anatomical mapping (FAM)**. The shell is, literally, the
envelope of where the catheter has been.

Two consequences follow, and both matter:

1. **Unvisited regions do not exist.** The mesh cannot represent anatomy the
   catheter never touched.
2. **The geometry and the electrical data are sampled at completely different
   densities.** FAM produces thousands of vertices; electrical points number in
   the hundreds. They are separate acquisitions.
""")

code("""
# MEASURED / DERIVED. Mesh resolution vs electrical sampling density.
points, triangles = case.points, case.indices

edges = np.vstack([
    triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]
])
edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)

tri_pts = points[triangles]
areas = 0.5 * np.linalg.norm(
    np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1
)

egm_points = electric.bipolar_egm.points
tree = spatial.cKDTree(egm_points)
nn_distance = tree.query(egm_points, k=2)[0][:, 1]

print(f"Mesh vertices            : {len(points):,}")
print(f"Mesh triangles           : {len(triangles):,}")
print(f"Total surface area       : {areas.sum():,.0f} mm^2")
print(f"Median mesh edge length  : {np.median(edge_lengths):.2f} mm   <- FAM resolution")
print()
print(f"Electrical points        : {len(egm_points):,}")
print(f"Median point spacing     : {np.median(nn_distance):.2f} mm   <- electrical resolution")
print()
print(f"Vertices per point       : {len(points) / len(egm_points):.1f}")
print(f"Area per electrical point: {areas.sum() / len(egm_points):.0f} mm^2")
""")

md(r"""
Read those two resolutions against each other. The **geometry** is resolved to
about a millimetre. The **electricity** is sampled every few millimetres, and
each point stands in for tens of square millimetres of tissue. The anatomy is
far better resolved than the physiology painted onto it — a mismatch the display
hides completely, because both are drawn on the same smooth surface.

### Where the catheter went: projection distance

Electrical points are recorded wherever the catheter tip was, which is near the
wall but not exactly on it. The system projects each point onto the mesh. That
projection distance is a quality signal: a point recorded far from the wall was
probably not in good contact.
""")

code("""
# MEASURED. Distance from recording position to its projection on the shell.
projection_distance = np.linalg.norm(
    electric.bipolar_egm.points - electric.surface.nearest_point, axis=1
)

print(f"median : {np.nanmedian(projection_distance):.2f} mm")
print(f"p95    : {np.nanpercentile(projection_distance, 95):.2f} mm")
print(f"max    : {np.nanmax(projection_distance):.2f} mm")

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(projection_distance, bins=50, color="#4c72b0", edgecolor="white", lw=0.4)
ax.axvline(np.nanmedian(projection_distance), color="crimson", ls="--",
           label=f"median {np.nanmedian(projection_distance):.2f} mm")
ax.set_xlabel("Distance from recording site to shell (mm)")
ax.set_ylabel("Points")
ax.set_title("How far each electrical point sat from the wall")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
""")
```

- [ ] **Step 2: Add free boundaries — the anatomy the holes represent**

```python
md(r"""
### The holes are anatomy

The shell is not a closed surface. Where the chamber opens into a vein or a
valve, the mesh simply ends, leaving a **free boundary** — an edge belonging to
only one triangle. Finding them is pure topology, and they identify real
anatomical structures.

For a left atrium we expect roughly six: the mitral valve (much the largest),
four pulmonary veins, and the left atrial appendage.
""")

code("""
# DERIVED, cross-checked against OpenEP.
from collections import Counter

edge_keys = Counter()
for tri in triangles:
    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        edge_keys[(min(a, b), max(a, b))] += 1

our_boundary_edges = [edge for edge, count in edge_keys.items() if count == 1]
print(f"our boundary edges  : {len(our_boundary_edges)}")

mesh = case.create_mesh()
from openep.mesh.mesh_routines import get_free_boundaries

free_boundaries = get_free_boundaries(mesh)
separated = free_boundaries.separate_boundaries()
lengths = free_boundaries.calculate_lengths()
boundary_areas = free_boundaries.calculate_areas()

print(f"OpenEP boundary edges : {sum(len(b) for b in separated)}")
print(f"distinct boundaries   : {len(separated)}")
print()
order = np.argsort(lengths)[::-1]
print(f"{'#':<3}{'edges':>7}{'perimeter mm':>15}{'area mm^2':>12}   likely structure")
for rank, idx in enumerate(order):
    guess = "mitral valve" if rank == 0 else "pulmonary vein / appendage"
    print(f"{rank:<3}{len(separated[idx]):>7}{lengths[idx]:>15.1f}{boundary_areas[idx]:>12.1f}   {guess}")
""")

md(r"""
Our raw edge count matches OpenEP's, and the structure is exactly what the anatomy
predicts: one large opening — the mitral valve, at roughly 1,077 mm² — and five
smaller ones for the pulmonary veins and appendage. **We identified cardiac
anatomy from nothing but mesh connectivity.**
""")
```

- [ ] **Step 3: Add the first 3D render, both engines**

```python
md(r"""
### The first render

Bare anatomy, no physiology yet. Two engines, as promised: PyVista gives a live
VTK view while the kernel runs, Plotly stays embedded in the saved file.
""")

code("""
# Live PyVista/trame view.
epkit.show_pyvista(mesh, color="#c8a0a0", title="Left atrium — FAM shell")
""")

code("""
# Embedded Plotly view, with the free boundaries traced on top.
shell = mesh_to_plotly(points, triangles, name="atrial shell", opacity=1.0)

boundary_traces = []
palette = ["#e45756", "#4c78a8", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]
for rank, idx in enumerate(order):
    edge_indices = separated[idx]
    ring = np.append(edge_indices[:, 0], edge_indices[0, 0])
    ring_points = free_boundaries.points[ring] if hasattr(free_boundaries, "points") else points[ring]
    boundary_traces.append(
        go.Scatter3d(
            x=ring_points[:, 0], y=ring_points[:, 1], z=ring_points[:, 2],
            mode="lines", line=dict(width=6, color=palette[rank % len(palette)]),
            name=f"boundary {rank} ({lengths[idx]:.0f} mm)",
        )
    )

plotly_figure([shell, *boundary_traces],
              title="Atrial shell with free boundaries (veins and valve)").show()
""")
```

- [ ] **Step 4: Rebuild, execute, verify**

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ep --ExecutePreprocessor.timeout=900 ep_primer.ipynb
```

Expected: no exceptions; 6 distinct boundaries reported; largest area ≈1,077 mm²; both renders appear.

If the `Scatter3d` ring indexing raises, print `separated[idx].shape` and `free_boundaries.points.shape` and index whichever array the edge indices actually refer to — do not silently drop the boundary traces.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_notebook.py ep_primer.ipynb
git commit -m "feat(notebook): Part 3 — geometry, FAM resolution, free boundaries, first 3D render"
```

---

### Task 7: Notebook Parts 4-5 — interpolation honesty and the activation map

**Files:**
- Modify: `scripts/build_notebook.py`, `ep_primer.ipynb`

**Interfaces:**
- Consumes: `case`, `mesh`, `lat`, `valid`, `points`, `triangles`
- Produces: `vertex_lat` (interpolated per-vertex LAT) used by Part 7

- [ ] **Step 1: Add Part 4 — the interpolation problem and its honesty check**

```python
md(r"""
## Part 4 — From points to maps: the interpolation problem

This is the heart of the matter. We have **1,366 measurements** and **9,279
vertices** to colour. Every vertex that was not measured gets a value invented by
an interpolator.

The standard approach is inverse-distance or radial-basis weighting: a vertex
takes a weighted average of nearby measurements, weighted by proximity. Crucially
it is also **distance-masked** — beyond some radius from any measurement, the
interpolator refuses to guess and returns NaN, leaving a hole in the map.

That refusal is the most honest thing the algorithm does.
""")

code("""
# DERIVED. Inverse-distance interpolation with an explicit distance mask.
def interpolate_to_surface(sample_points, sample_values, target_points,
                           max_distance=10.0, k=8, power=2.0):
    \"\"\"Inverse-distance weighted interpolation with a hard distance cutoff.

    Vertices with no sample within `max_distance` return NaN rather than a
    fabricated value.
    \"\"\"
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
    weights[~np.isfinite(weights)] = 1e12          # exact hits dominate

    values = np.sum(weights * sample_values[indices], axis=1) / np.sum(weights, axis=1)
    values[distances[:, 0] > max_distance] = np.nan
    return values


surface_points = electric.surface.nearest_point
vertex_lat = interpolate_to_surface(surface_points[valid], lat[valid], points)

print(f"vertices coloured : {np.isfinite(vertex_lat).sum():,} of {len(points):,}")
print(f"left as NaN       : {np.isnan(vertex_lat).mean() * 100:.1f}%")
print(f"OpenEP's own map  : {np.isnan(case.fields.local_activation_time).mean() * 100:.1f}% NaN")
""")

md(r"""
Our NaN fraction lands close to OpenEP's, from an independent implementation.
Roughly **two of every five vertices cannot be honestly coloured at all**.

### How much of the map is real?

Now the uncomfortable experiment. Same measurements, different reasonable
choices — and the resulting maps differ.
""")

code("""
# DERIVED. The same data under three interpolation regimes.
from scipy.interpolate import griddata

variants = {
    "IDW, 10 mm mask": vertex_lat,
    "IDW, 30 mm mask": interpolate_to_surface(
        surface_points[valid], lat[valid], points, max_distance=30.0),
    "Nearest neighbour": griddata(
        surface_points[valid], lat[valid], points, method="nearest"),
}

for name, field in variants.items():
    coloured = np.isfinite(field).mean() * 100
    print(f"{name:<20} coloured {coloured:5.1f}%   "
          f"range {np.nanmin(field):6.1f} to {np.nanmax(field):6.1f} ms")

pairs = list(variants.items())
for (name_a, a), (name_b, b) in [(pairs[0], pairs[1]), (pairs[0], pairs[2])]:
    both = np.isfinite(a) & np.isfinite(b)
    print(f"\\nmedian |{name_a} - {name_b}| = {np.nanmedian(np.abs(a[both] - b[both])):.1f} ms")
""")

code("""
traces = []
for offset, (name, field) in enumerate(variants.items()):
    shifted = points.copy()
    shifted[:, 0] += offset * 145.0
    traces.append(mesh_to_plotly(
        shifted, triangles, scalars=field, colorscale="Turbo",
        cmin=np.nanmin(lat), cmax=np.nanmax(lat),
        name=name, showscale=(offset == 0), colorbar_title="LAT (ms)",
    ))

plotly_figure(traces, title="Same measurements, three interpolators — "
                           "left: 10 mm mask | centre: 30 mm mask | right: nearest").show()
""")

md(r"""
The left-hand map admits what it does not know. The middle one fills those holes
by extrapolating up to 30 mm from the nearest real measurement — smoother,
prettier, and partly fiction. The right-hand one is confident everywhere and
honest nowhere.

**None of these is more "correct" than the others.** They encode different
assumptions about how far a measurement's influence should reach. A clinician
reading a map is reading those assumptions as much as the tissue.

### Watching the map dissolve

What happens as measurements are thrown away?
""")

code("""
# DERIVED. Degrade the sampling density and watch the map degrade with it.
rng = np.random.default_rng(42)
valid_indices = np.flatnonzero(valid)

traces = []
for offset, n_keep in enumerate([len(valid_indices), 400, 150, 50]):
    keep = rng.choice(valid_indices, size=min(n_keep, len(valid_indices)), replace=False)
    field = interpolate_to_surface(surface_points[keep], lat[keep], points)

    shifted = points.copy()
    shifted[:, 0] += offset * 145.0
    traces.append(mesh_to_plotly(
        shifted, triangles, scalars=field, colorscale="Turbo",
        cmin=np.nanmin(lat), cmax=np.nanmax(lat),
        name=f"{len(keep)} points", showscale=(offset == 0), colorbar_title="LAT (ms)",
    ))
    print(f"{len(keep):>5} points -> {np.isfinite(field).mean() * 100:5.1f}% coloured")

plotly_figure(traces, title="1366 -> 400 -> 150 -> 50 points: how map confidence decays").show()
""")

md(r"""
The overall pattern survives surprisingly far down. **Fine structure does not** —
and fine structure is exactly what identifies an arrhythmia circuit. A sparsely
sampled map can look perfectly convincing while having lost the thing you are
looking for.
""")
```

- [ ] **Step 2: Add Part 5 — the activation map**

```python
md(r"""
## Part 5 — The activation map

Now the payoff. Colour the shell by **when** each region activated, and the
propagation pattern of the heartbeat becomes visible.

Convention: **red is early, blue is late.**
""")

code("""
# DERIVED. Activation timing across the chamber.
finite_lat = vertex_lat[np.isfinite(vertex_lat)]
earliest_vertex = int(np.nanargmin(vertex_lat))
latest_vertex = int(np.nanargmax(vertex_lat))

print(f"Earliest activation : {np.nanmin(vertex_lat):.0f} ms")
print(f"Latest activation   : {np.nanmax(vertex_lat):.0f} ms")
print(f"Total activation    : {np.nanmax(vertex_lat) - np.nanmin(vertex_lat):.0f} ms")
print(f"Measured cycle len  : approximately {np.nanmax(lat) - np.nanmin(lat):.0f} ms across points")
""")

code("""
epkit.show_pyvista(
    mesh, scalars=vertex_lat, cmap="turbo_r",
    clim=(np.nanmin(vertex_lat), np.nanmax(vertex_lat)),
    title="Local activation time (ms)",
)
""")

code("""
activation = mesh_to_plotly(
    points, triangles, scalars=vertex_lat, colorscale="Turbo_r",
    colorbar_title="LAT (ms)", name="activation",
)

extremes = go.Scatter3d(
    x=points[[earliest_vertex, latest_vertex], 0],
    y=points[[earliest_vertex, latest_vertex], 1],
    z=points[[earliest_vertex, latest_vertex], 2],
    mode="markers+text",
    marker=dict(size=9, color=["red", "blue"]),
    text=["earliest", "latest"], textposition="top center", name="extremes",
)

plotly_figure([activation, extremes], title="Activation map — red early, blue late").show()
""")

md(r"""
### Reading it

- **Focal pattern** — activation radiates outward from one early site, like a
  stone dropped in water. The earliest point is the source, and it is the ablation
  target.
- **Macro-reentrant pattern** — the earliest and latest regions sit *adjacent* to
  each other, with colour sweeping through the full cycle in between. Early meets
  late because the wave never stops; it circulates. The target is not a point but
  the narrow isthmus the circuit depends on.

The distinction is the central diagnostic question in an ablation procedure, and
it is answered by looking at this picture.

Total activation time carries the same information numerically: if it approaches
the full cycle length, the chamber is never electrically quiet, which means the
wavefront is continuously circulating — reentry.
""")

code("""
# DERIVED. Isochrones make crowding visible: tight bands mean slow conduction.
isochrone_step = 10  # ms
banded = np.floor(vertex_lat / isochrone_step) * isochrone_step

plotly_figure(
    [mesh_to_plotly(points, triangles, scalars=banded, colorscale="Turbo_r",
                    colorbar_title=f"LAT ({isochrone_step} ms bands)")],
    title=f"Isochrone map — {isochrone_step} ms bands. Crowded bands = slow conduction",
).show()
""")
```

- [ ] **Step 3: Rebuild, execute, verify**

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ep --ExecutePreprocessor.timeout=1200 ep_primer.ipynb
```

Expected: our NaN fraction prints within a few percent of OpenEP's 41.5%; all comparison renders appear.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_notebook.py ep_primer.ipynb
git commit -m "feat(notebook): Parts 4-5 — interpolation honesty check and activation map"
```

---

### Task 8: Notebook Parts 6-7 — voltage substrate and conduction velocity

**Files:**
- Modify: `scripts/build_notebook.py`, `ep_primer.ipynb`

**Interfaces:**
- Consumes: `case`, `vertex_lat`, `points`, `triangles`, `epkit.conduction_velocity`, `epkit.voltage_colorscale`
- Produces: `vertex_voltage`, `cv_speed`, `cv_direction`

- [ ] **Step 1: Add Part 6 — voltage, scar, and bipolar's direction dependence**

```python
md(r"""
## Part 6 — Voltage: mapping the substrate

Timing tells you how the wave moved. **Amplitude tells you what the tissue is.**

Healthy myocardium depolarises in bulk and synchronously, producing a large sharp
bipolar deflection. Scar has fewer surviving myocytes, so it generates less
signal. Bipolar peak-to-peak voltage therefore acts as a proxy for viable tissue,
and the clinical thresholds are well established for atrial tissue:

| Bipolar voltage | Interpretation |
|---|---|
| < 0.5 mV | dense scar |
| 0.5 - 1.5 mV | border zone / diseased |
| > 1.5 mV | healthy myocardium |

These are conventions calibrated against imaging and outcomes, not laws of
physics — but they drive real clinical decisions.
""")

code("""
# DERIVED, cross-checked against the system's stored values.
reference_annotation = annotations.reference_activation_time

our_voltage = np.full(electric.n_points, np.nan)
for i in range(electric.n_points):
    start = int(reference_annotation[i] + woi[i, 0])
    stop = int(reference_annotation[i] + woi[i, 1])
    start, stop = max(start, 0), min(stop, electric.n_samples)
    if stop - start < 2:
        continue
    segment = stored_bipolar[i, start:stop]
    our_voltage[i] = np.nanmax(segment) - np.nanmin(segment)

stored_voltage = electric.bipolar_egm.voltage
both = np.isfinite(our_voltage) & np.isfinite(stored_voltage)

print(f"our peak-to-peak    : {np.nanmedian(our_voltage):.3f} mV median")
print(f"system's value      : {np.nanmedian(stored_voltage):.3f} mV median")
print(f"correlation         : {np.corrcoef(our_voltage[both], stored_voltage[both])[0, 1]:.4f}")
print(f"median |difference| : {np.nanmedian(np.abs(our_voltage[both] - stored_voltage[both])):.3f} mV")
""")

code("""
# DERIVED. Interpolate voltage onto the shell and classify tissue.
vertex_voltage = interpolate_to_surface(surface_points, stored_voltage, points)

dense_scar = vertex_voltage < 0.5
border = (vertex_voltage >= 0.5) & (vertex_voltage < 1.5)
healthy = vertex_voltage >= 1.5

vertex_area = np.zeros(len(points))
np.add.at(vertex_area, triangles.ravel(), np.repeat(areas / 3.0, 3))
total_area = vertex_area[np.isfinite(vertex_voltage)].sum()

print(f"{'category':<16}{'vertices':>10}{'area mm^2':>12}{'share':>9}")
for label, mask in [("dense scar", dense_scar), ("border zone", border), ("healthy", healthy)]:
    area = vertex_area[mask & np.isfinite(vertex_voltage)].sum()
    print(f"{label:<16}{int(mask.sum()):>10}{area:>12,.0f}{area / total_area * 100:>8.1f}%")
""")

code("""
voltage_map = mesh_to_plotly(
    points, triangles, scalars=np.clip(vertex_voltage, 0, 5.0),
    colorscale=epkit.voltage_colorscale(), cmin=0.0, cmax=5.0,
    colorbar_title="Bipolar (mV)", name="voltage",
)
plotly_figure([voltage_map],
              title="Voltage map — red < 0.5 mV dense scar, purple > 1.5 mV healthy").show()
""")

md(r"""
### The catch: bipolar voltage depends on wavefront direction

Part 2 established that a bipolar signal is the difference between two electrodes
— a spatial derivative along the axis joining them. A derivative along an axis is
maximal when the wavefront travels **along** that axis and vanishes when it
travels **perpendicular** to it.

So identical healthy tissue can read as low voltage purely because of the angle
the catheter happened to sit at. Some low-voltage regions on any bipolar map are
not scar; they are geometry. The dataset stores the electrode orientation, so we
can look for the effect directly.
""")

code("""
# MEASURED / DERIVED. Voltage against electrode orientation.
raw = scipy.io.loadmat(str(epkit.DATA_DIR / "openep_dataset_1.mat"))
bar_direction = raw["userdata"][0, 0]["electric"][0, 0]["barDirection"]
bar_direction = np.asarray(bar_direction, dtype=float)

normals = electric.surface.normals
alignment = np.abs(np.sum(bar_direction * normals, axis=1))
alignment /= (np.linalg.norm(bar_direction, axis=1) * np.linalg.norm(normals, axis=1))

usable = np.isfinite(alignment) & np.isfinite(stored_voltage)
lower, upper = np.nanpercentile(alignment[usable], [33, 67])
groups = {
    "in-plane (aligned)": alignment[usable] <= lower,
    "intermediate": (alignment[usable] > lower) & (alignment[usable] < upper),
    "out-of-plane": alignment[usable] >= upper,
}
for label, mask in groups.items():
    values = stored_voltage[usable][mask]
    print(f"{label:<22} n={mask.sum():>4}   median {np.nanmedian(values):.3f} mV")
""")

md(r"""
Whatever separation appears here, the mechanism is real and well documented: it is
why unipolar voltage — which has no directional axis — is often read alongside
bipolar, and why bipolar low-voltage areas are confirmed rather than trusted
outright.
""")
```

- [ ] **Step 2: Add Part 7 — conduction velocity**

```python
md(r"""
## Part 7 — Conduction velocity

We have a scalar timing field across a surface. Its **gradient** tells us how
fast and in which direction the wave moved.

If activation time is $t(\mathbf{x})$ in ms and position is in mm, then
$|\nabla t|$ has units ms/mm — the time cost per millimetre travelled. Speed is
its reciprocal:

$$\mathrm{CV} = \frac{1}{|\nabla t|}$$

with the wave travelling along $+\nabla t$, toward later activation. On a
triangulated surface the gradient is computed per triangle: a linear field over
a triangle has a constant gradient, obtained by solving the two edge equations
together with the constraint that the gradient lies in the triangle plane. We
then average onto vertices.

`openep-py` has **no conduction-velocity function at all**, so there is nothing
to check against. Instead the implementation is validated in the test suite
against a synthetic planar wave with a known answer: a wave at 1 mm/ms must come
back as exactly 1000 mm/s. That test passes, so the code below is trustworthy.

Normal atrial conduction is roughly **600-1200 mm/s**. Below ~300 mm/s is slow
conduction — the substrate that sustains reentry.
""")

code("""
# DERIVED. Conduction velocity from the interpolated activation field.
cv_speed, cv_direction = epkit.conduction_velocity(points, triangles, vertex_lat)

finite_cv = cv_speed[np.isfinite(cv_speed)]
print(f"vertices with CV : {finite_cv.size:,}")
print(f"median CV        : {np.median(finite_cv):,.0f} mm/s")
print(f"IQR              : {np.percentile(finite_cv, 25):,.0f} to {np.percentile(finite_cv, 75):,.0f} mm/s")
print(f"slow (<300 mm/s) : {np.mean(finite_cv < 300) * 100:.1f}% of mapped surface")

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(np.clip(finite_cv, 0, 3000), bins=60, color="#55a868", edgecolor="white", lw=0.4)
ax.axvline(300, color="crimson", ls="--", label="300 mm/s — slow conduction")
ax.set_xlabel("Conduction velocity (mm/s)")
ax.set_ylabel("Vertices")
ax.set_title("Conduction velocity distribution")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
""")

code("""
cv_display = np.clip(cv_speed, 0, 2000)
cv_map = mesh_to_plotly(
    points, triangles, scalars=cv_display, colorscale="Viridis",
    cmin=0, cmax=2000, colorbar_title="CV (mm/s)", name="conduction velocity",
)

step = max(len(points) // 400, 1)
sample = np.arange(0, len(points), step)
sample = sample[np.isfinite(cv_direction[sample]).all(axis=1)]
arrow_scale = 3.0

arrows = go.Cone(
    x=points[sample, 0], y=points[sample, 1], z=points[sample, 2],
    u=cv_direction[sample, 0] * arrow_scale,
    v=cv_direction[sample, 1] * arrow_scale,
    w=cv_direction[sample, 2] * arrow_scale,
    sizemode="absolute", sizeref=4.0, showscale=False,
    colorscale=[[0, "black"], [1, "black"]], name="propagation",
)

plotly_figure([cv_map, arrows],
              title="Conduction velocity with propagation direction").show()
""")

md(r"""
### When conduction velocity lies

CV is the most derived quantity in this notebook — a numerical gradient of an
interpolated field built from annotated timings. Every one of those layers can
inject error, and three situations produce confidently wrong numbers:

- **Wave collision.** Where two wavefronts meet, timing has a genuine
  discontinuity. The gradient there is huge and the computed CV drops toward
  zero, mimicking slow conduction that does not exist.
- **Breakthrough.** Where a wave emerges from a different plane, the surface
  gradient near-vanishes and CV goes implausibly high.
- **Sparse sampling.** Where the interpolator invented the field, the gradient
  describes the interpolator, not the tissue.

Cross-reference the CV map against the point-density and voltage maps before
believing any specific slow-conduction region.
""")
```

- [ ] **Step 3: Rebuild, execute, verify**

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ep --ExecutePreprocessor.timeout=1500 ep_primer.ipynb
```

Expected: voltage correlation with the system > 0.9; median CV in the hundreds-to-low-thousands mm/s range. If median CV is wildly outside 100-3000 mm/s, check the mm/ms → mm/s factor of 1000 before adjusting anything else.

Add `import scipy.io` to the Part 0 imports cell if the `barDirection` cell reports a NameError.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_notebook.py ep_primer.ipynb
git commit -m "feat(notebook): Parts 6-7 — voltage substrate and conduction velocity"
```

---

### Task 9: Notebook Parts 8-10 — ablation, tagging, workflow, finale

**Files:**
- Modify: `scripts/build_notebook.py`, `ep_primer.ipynb`

**Interfaces:**
- Consumes: everything prior; `epkit.load_rfindex`, `epkit.fractionation_index`
- Produces: the finished notebook

- [ ] **Step 1: Add Part 8 — RF delivery, lesion tags, and tagging**

```python
md(r"""
## Part 8 — Ablation: delivering and recording therapy

Diagnosis identifies the target; **ablation** destroys it. A radiofrequency
catheter passes current into the tissue, heating it resistively. Above roughly
50 °C myocytes die and are replaced by scar, which is electrically inert — and
that is the point. A well-placed line of lesions blocks the circuit.

The export records the delivery in detail.
""")

code("""
# MEASURED. The physical record of therapy.
ablation = case.ablation
force = ablation.force

print(f"RF samples     : {len(ablation.times):,}")
print(f"Force samples  : {len(force.times):,}")
print(f"Power          : {np.nanmin(ablation.power):.0f} to {np.nanmax(ablation.power):.0f} W")
print(f"Temperature    : {np.nanmin(ablation.temperature):.0f} to {np.nanmax(ablation.temperature):.0f} C")
print(f"Impedance      : {np.nanmin(ablation.impedance):.0f} to {np.nanmax(ablation.impedance):.0f} ohm")
print(f"Contact force  : {np.nanmin(force.force):.0f} to {np.nanmax(force.force):.0f} g")
""")

code("""
# MEASURED. One window of RF delivery, all four channels.
active = np.flatnonzero(np.asarray(ablation.power) > 5)
start = active[0]
stop = min(start + 400, len(ablation.times))
window = slice(start, stop)
t = (np.asarray(ablation.times)[window] - ablation.times[start]) / 1000.0

fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
axes[0].plot(t, np.asarray(ablation.power)[window], color="#c44e52")
axes[0].set_ylabel("Power (W)")
axes[0].set_title("A single RF application")
axes[1].plot(t, np.asarray(ablation.temperature)[window], color="#dd8452")
axes[1].set_ylabel("Temp (C)")
axes[2].plot(t, np.asarray(ablation.impedance)[window], color="#4c72b0")
axes[2].set_ylabel("Impedance (ohm)")
axes[2].set_xlabel("Time (s)")
for ax in axes:
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
""")

md(r"""
Watch the impedance. It **falls** as the lesion forms — heated tissue conducts
electricity better. That drop is one of the few real-time signals that the lesion
is actually forming, rather than the catheter merely delivering power into blood.

### Lesion tags: what the operator left behind

Each completed lesion is recorded as a **tag** — a marked position with the
metrics describing lesion quality. `openep-py` does not load these (its
documentation says so explicitly), so we parse them ourselves.
""")

code("""
# MEASURED. The 90 ablation lesion tags, parsed directly from rfindex.
tags = epkit.load_rfindex(1)

print(f"lesion tags : {len(tags)}")
print(f"index metric: {tags.index_name}")
print()
print(f"{'metric':<22}{'median':>10}{'min':>10}{'max':>10}")
for label, values, unit in [
    ("duration", tags.duration, "s"),
    ("average force", tags.avg_force, "g"),
    ("max power", tags.max_power, "W"),
    ("max temperature", tags.max_temp, "C"),
    ("force-time integral", tags.fti, "gs"),
]:
    print(f"{label + ' (' + unit + ')':<22}{np.nanmedian(values):>10.1f}"
          f"{np.nanmin(values):>10.1f}{np.nanmax(values):>10.1f}")
""")

md(r"""
**Force-time integral (FTI)** is the product of contact force and duration —
grams-seconds. It exists because lesion depth depends on how hard the catheter
pressed and for how long, not merely on power. Poor contact means a superficial
lesion that looks identical on the map and recovers weeks later.

That recovery is the central failure mode of atrial fibrillation ablation:
pulmonary vein reconnection. Roughly a fifth to a third of patients need a repeat
procedure, and reconnected gaps in an ablation line are the usual reason. FTI and
the proprietary lesion indices exist to predict which lesions will hold.
""")

code("""
# MEASURED. Lesion tags on the anatomy, coloured by force-time integral.
shell = mesh_to_plotly(points, triangles, name="shell", opacity=0.55, showscale=False)

lesions = go.Scatter3d(
    x=tags.points[:, 0], y=tags.points[:, 1], z=tags.points[:, 2],
    mode="markers",
    marker=dict(
        size=7, color=tags.fti, colorscale="Inferno", showscale=True,
        colorbar=dict(title="FTI (gs)", thickness=14, len=0.7),
        line=dict(width=0.5, color="white"),
    ),
    text=[f"lesion {i}: {f:.0f} gs, {d:.1f} s, {p:.0f} W"
          for i, (f, d, p) in enumerate(zip(tags.fti, tags.duration, tags.max_power))],
    hoverinfo="text", name="lesions",
)

plotly_figure([shell, lesions],
              title=f"{len(tags)} ablation lesions, coloured by force-time integral").show()
""")

md(r"""
### Tagging in general

"Tag" covers three different things in an EP lab, and it is worth separating them:

| Kind | What it marks | In this data |
|---|---|---|
| **Ablation tags** | where therapy was delivered | **90, with full metrics** |
| **Anatomical / landmark tags** | vein ostia, valve, phrenic nerve capture | `landmark_points` field exists |
| **Point tags** | electrogram features: fractionated, double potential, late potential | field exists, **empty in this export** |

That last row is the honest gap. `electric.tags` is present in the OpenEP schema
and empty in both datasets — the operator's annotations did not survive the
export. Rather than skip the topic, we can detect the features ourselves.

**Fractionated** electrograms show many small deflections instead of one sharp
spike, marking slow disorganised conduction through surviving muscle strands
inside scar. Those regions are ablation targets in their own right.
""")

code("""
# DERIVED. Detect fractionation ourselves, since the operator's tags are absent.
window_samples = (int(reference_annotation[0] + woi[0, 0]),
                  int(reference_annotation[0] + woi[0, 1]))
deflections = epkit.fractionation_index(stored_bipolar, electric.frequency, window_samples)

complex_points = deflections >= np.nanpercentile(deflections, 90)
print(f"deflection count: median {np.nanmedian(deflections):.1f}, "
      f"90th percentile {np.nanpercentile(deflections, 90):.1f}")
print(f"flagged as complex: {int(complex_points.sum())} of {len(deflections)} points")

simple_idx = int(np.nanargmin(np.where(np.isfinite(deflections), deflections, np.inf)))
complex_idx = int(np.nanargmax(np.where(np.isfinite(deflections), deflections, -np.inf)))

fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
sl = slice(*window_samples)
axes[0].plot(times_ms[sl], stored_bipolar[simple_idx, sl], color="#2ca02c", lw=1.0)
axes[0].set_title(f"Sharp / organised — {deflections[simple_idx]:.0f} deflections")
axes[1].plot(times_ms[sl], stored_bipolar[complex_idx, sl], color="#d62728", lw=1.0)
axes[1].set_title(f"Fractionated — {deflections[complex_idx]:.0f} deflections")
axes[1].set_xlabel("Time (ms)")
for ax in axes:
    ax.set_ylabel("mV")
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
""")

code("""
# DERIVED. Our own complex-point tags on the anatomy.
flagged = complex_points & np.isfinite(surface_points).all(axis=1)
voltage_shell = mesh_to_plotly(
    points, triangles, scalars=np.clip(vertex_voltage, 0, 5.0),
    colorscale=epkit.voltage_colorscale(), cmin=0, cmax=5.0,
    colorbar_title="Bipolar (mV)", opacity=0.9,
)
flags = go.Scatter3d(
    x=surface_points[flagged, 0], y=surface_points[flagged, 1], z=surface_points[flagged, 2],
    mode="markers", marker=dict(size=5, color="black", symbol="x"),
    name="derived complex points",
)
plotly_figure([voltage_shell, flags],
              title="Derived complex-point tags over the voltage map").show()
""")
```

- [ ] **Step 2: Add Part 9 — workflow, acquisition modes, PASO**

```python
md(r"""
## Part 9 — The workflow this data came from

The arrays we have been reading are the residue of a procedure that runs roughly
like this:

1. **Access** — catheters to the chamber; for the left atrium, across the septum.
2. **Geometry (FAM)** — sweep the chamber to build the shell (Part 3).
3. **Mapping** — collect electrograms across the surface (Parts 2, 4, 5, 6).
4. **Diagnosis** — read the activation and voltage maps; focal or reentrant?
5. **Pace mapping / entrainment** — confirm the mechanism and target.
6. **Ablation** — deliver lesions, tag them (Part 8).
7. **Validation** — remap to prove the target is gone, and wait to check durability.

### Point-by-point vs continuous acquisition

Two ways to collect step 3.

**Point-by-point** — position the catheter, confirm contact and stability, accept
one point. Slow, deliberate, every point reviewed by a human. Hundreds of points.

**Continuous / pattern acquisition** — a multipolar catheter with many electrodes
sweeps while the system automatically accepts everything meeting quality criteria.
Thousands to tens of thousands of points, no human reviewing each one.

The acquisition mode is not stored in the export, but it leaves fingerprints.
""")

code("""
# DERIVED. Fingerprints of how the data was collected.
catheter_names = raw["userdata"][0, 0]["electric"][0, 0]["electrodeNames_bip"]
labels = [str(np.asarray(n).ravel()[0]) if np.asarray(n).size else "" for n in catheter_names.ravel()]
unique_labels, counts = np.unique(labels, return_counts=True)

print("Electrode channels contributing points:")
for label, count in sorted(zip(unique_labels, counts), key=lambda p: -p[1]):
    print(f"  {label:<14}{count:>6} points")

print(f"\\ndistinct electrode pairs : {len(unique_labels)}")
print(f"points per pair          : {np.median(counts):.0f} median")
print(f"total points             : {electric.n_points}")
print(f"median nearest-neighbour : {np.median(nn_distance):.2f} mm")
""")

md(r"""
Ten bipolar channels named `20A_1-2` through `20A_19-20`: a **20-pole circular
mapping catheter**, collecting through ten electrode pairs at once. With 1,366
points spread fairly evenly across those ten channels, this was multi-electrode
acquisition — not one-point-at-a-time. That is the acquisition mode, recovered
from electrode naming alone.

### Pace mapping and PASO

When activation mapping cannot localise a focal source — often because the
arrhythmia is infrequent and will not stay running long enough to map — the logic
gets inverted. Instead of recording where the wave went, **pace from a candidate
site and see whether the resulting body-surface ECG matches the clinical
arrhythmia.**

If pacing from a site reproduces the arrhythmia's 12-lead morphology, that site
activates the heart the same way the arrhythmia does, so it is at or very near the
source. **PASO** is CARTO's module for this: it correlates the paced 12-lead
complex against a stored template and reports a percentage, typically calling
> 95% across all twelve leads a good match.

**This dataset cannot reproduce PASO.** It stores a *single* ECG channel, not
twelve, and no paced templates. What follows demonstrates the correlation
mathematics on the one lead available — the method, not the clinical tool.
""")

code("""
# ILLUSTRATIVE. Normalised cross-correlation of ECG morphology.
# The method behind PASO, on one lead instead of twelve. Not a PASO result.
ecg = electric.ecg.ecg[:, :, 0]

def normalised_correlation(a, b):
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denominator = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
    return np.nansum(a * b) / denominator if denominator > 0 else np.nan

template_idx = 0
segment = slice(*window_samples)
template = ecg[template_idx, segment]

scores = np.array([normalised_correlation(template, ecg[i, segment])
                   for i in range(min(400, len(ecg)))])

print(f"template          : point {electric.internal_names[template_idx]}")
print(f"beats compared    : {scores.size}")
print(f"median correlation: {np.nanmedian(scores):.3f}")
print(f"above 0.95        : {np.nanmean(scores > 0.95) * 100:.1f}%")
print()
print("In real PASO this runs across all 12 leads simultaneously, and a")
print("pacing site scoring >95% on every lead is taken as the arrhythmia origin.")

fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(scores[np.isfinite(scores)], bins=40, color="#8172b3", edgecolor="white", lw=0.4)
ax.axvline(0.95, color="crimson", ls="--", label="0.95 match threshold")
ax.set_xlabel("Normalised correlation with template")
ax.set_ylabel("Beats")
ax.set_title("Template correlation — the mathematics behind pace mapping (single lead)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
""")

md(r"""
### What can and cannot be reconstructed

**Solid ground — measured directly:**

- Chamber geometry, to about a millimetre, wherever the catheter went.
- Electrogram waveforms at recorded sites, at 1 kHz.
- Activation timing at those sites, subject to the annotation choice.
- Signal amplitude, and therefore a tissue-health proxy.
- Everything about the ablation: position, power, force, temperature, impedance.

**Inference — derived, with real uncertainty:**

- Any value at a vertex that was not measured, which is most of them.
- Conduction velocity: a gradient of an interpolation of annotated timings.
- Scar classification, resting on thresholds and on direction-dependent bipolar
  voltage.
- The arrhythmia mechanism, read from a map built on all of the above.

**Out of reach from this data entirely:**

- Anything transmural. This is a *surface* map; the wall has thickness, and
  circuits can run through it, invisible here.
- Anatomy the catheter never visited.
- True 12-lead pace mapping, and the operator's own point tags.
- What happens after the procedure — durability needs follow-up, not this file.
""")
```

- [ ] **Step 3: Add Part 10 — the composite finale**

```python
md(r"""
## Part 10 — Everything at once

Anatomy, activation, substrate, conduction, and therapy — the same chamber under
five different questions. This is what an electrophysiologist is holding in their
head simultaneously while deciding where to ablate.
""")

code("""
# The full reconstruction, side by side.
spacing = 150.0
panels = [
    ("Anatomy", None, "Greys", None, None),
    ("Activation (ms)", vertex_lat, "Turbo_r", np.nanmin(vertex_lat), np.nanmax(vertex_lat)),
    ("Voltage (mV)", np.clip(vertex_voltage, 0, 5), epkit.voltage_colorscale(), 0, 5),
    ("Conduction (mm/s)", np.clip(cv_speed, 0, 2000), "Viridis", 0, 2000),
]

traces = []
for offset, (title, field, scale, cmin, cmax) in enumerate(panels):
    shifted = points.copy()
    shifted[:, 0] += offset * spacing
    traces.append(mesh_to_plotly(
        shifted, triangles, scalars=field, colorscale=scale,
        cmin=cmin, cmax=cmax, name=title, showscale=False,
    ))

lesion_panel = go.Scatter3d(
    x=tags.points[:, 0] + 3 * spacing, y=tags.points[:, 1], z=tags.points[:, 2],
    mode="markers", marker=dict(size=5, color="red"), name="lesions",
)

figure = plotly_figure(
    [*traces, lesion_panel],
    title="Anatomy | Activation | Voltage | Conduction velocity + lesions",
    height=700,
)
figure.show()
""")

code("""
# One interactive view: substrate underneath, lesions on top, propagation arrows.
final = plotly_figure(
    [
        mesh_to_plotly(points, triangles, scalars=np.clip(vertex_voltage, 0, 5),
                       colorscale=epkit.voltage_colorscale(), cmin=0, cmax=5,
                       colorbar_title="Bipolar (mV)", opacity=0.95),
        go.Scatter3d(
            x=tags.points[:, 0], y=tags.points[:, 1], z=tags.points[:, 2],
            mode="markers",
            marker=dict(size=6, color=tags.fti, colorscale="Inferno",
                        line=dict(width=0.5, color="white")),
            name="ablation lesions",
            text=[f"{f:.0f} gs" for f in tags.fti], hoverinfo="text",
        ),
        arrows,
    ],
    title="The complete picture: substrate, therapy, and propagation",
    height=720,
)
final.show()
""")

md(r"""
## Where to go next

- **[OpenEP](https://openep.io)** — the project these datasets come from; the
  MATLAB implementation is more complete than the Python port.
- **openCARP** — simulate propagation rather than measure it, and test what a
  circuit *would* do.
- **The gaps in this notebook.** Omnipolar mapping addresses the bipolar
  direction-dependence of Part 6. Charge-density and non-contact mapping avoid the
  contact requirement entirely. Ripple mapping abandons annotation — the choice
  Part 2 showed moves the map — by animating raw voltage instead.

**The one thing worth keeping.** Every clinical EP map is a small number of real
measurements, interpolated across a surface, filtered through an annotation
choice, and rendered in colours that imply far more certainty than the underlying
data supports. Knowing which parts are measured and which are inferred is the
whole skill.
""")

write()
```

- [ ] **Step 4: Rebuild and execute the complete notebook**

```bash
cd /home/vraj/Documents/personal/ep
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ep --ExecutePreprocessor.timeout=2400 ep_primer.ipynb
```

Expected: full run, no exceptions, 90 lesion tags reported.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_notebook.py ep_primer.ipynb
git commit -m "feat(notebook): Parts 8-10 — ablation, tagging, workflow, composite render"
```

---

### Task 10: Verification and delivery

**Files:**
- Modify: `README.md`, `ep_primer.ipynb`

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/vraj/Documents/personal/ep
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass. Record the count.

- [ ] **Step 2: Clean-kernel execution from scratch**

This is the real acceptance test: it proves the notebook runs for someone who has never run it.

```bash
cd /home/vraj/Documents/personal/ep
.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=ep --ExecutePreprocessor.timeout=2400 \
  --output /tmp/ep_verify.ipynb ep_primer.ipynb
echo "exit: $?"
```

Expected: exit 0.

- [ ] **Step 3: Verify the renders persisted in the saved file**

```bash
.venv/bin/python - <<'PY'
import json

notebook = json.load(open("/home/vraj/Documents/personal/ep/ep_primer.ipynb"))
code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
with_output = [c for c in code_cells if c.get("outputs")]
plotly = sum(
    1
    for c in code_cells
    for o in c.get("outputs", [])
    if "application/vnd.plotly.v1+json" in o.get("data", {})
)
errors = [
    o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"
]

print(f"code cells      : {len(code_cells)}")
print(f"with output     : {len(with_output)}")
print(f"plotly figures  : {plotly}")
print(f"errors          : {len(errors)}")
for e in errors:
    print("  ", e.get("ename"), e.get("evalue"))
PY
```

Expected: zero errors, and at least 8 embedded Plotly figures.

- [ ] **Step 4: Check the notebook size is reasonable**

```bash
du -h /home/vraj/Documents/personal/ep/ep_primer.ipynb
```

If it exceeds ~25 MB, apply `epkit.decimate(points, triangles, field, target_faces=8000)` to the Plotly traces in the multi-panel cells of Parts 4 and 10, rebuild, and re-execute. Leave the PyVista paths at full resolution.

- [ ] **Step 5: Final commit**

```bash
cd /home/vraj/Documents/personal/ep
git add -A
git commit -m "docs: verified full-notebook execution and embedded renders"
```

---

## Self-Review

**Spec coverage.** Environment and pins → Task 1. Data loading and `rfindex` → Task 2. Hand-rolled CV/fractionation/LAT cleaning with tests → Task 3. Dual rendering and clinical colour scales → Task 4. Parts 0-2 → Task 5. Part 3 including FAM resolution and free boundaries → Task 6. Parts 4-5 including the interpolation honesty check → Task 7. Parts 6-7 including bipolar direction dependence → Task 8. Parts 8-10 including lesion tags, tagging taxonomy, derived complex points, acquisition mode, PASO, and the composite → Task 9. Verification → Task 10. Every spec section maps to a task.

**Cross-checks against OpenEP** appear where the spec promised: bipolar derivation (Part 2), free boundaries (Part 3), interpolation NaN fraction (Part 4), voltage (Part 6). CV has no OpenEP equivalent and is instead validated against a synthetic planar wave in Task 3.

**Type consistency.** `load_case(which)` and `load_rfindex(which)` take the same integer argument throughout. `LesionTags` field names used in Task 9 (`points`, `fti`, `duration`, `max_power`, `avg_force`, `max_temp`, `index_name`) match the dataclass in Task 2. `conduction_velocity` returns `(speed, direction)` in Task 3 and is unpacked as such in Task 8. `mesh_to_plotly` keyword names are consistent across Tasks 4, 6, 7, 8, 9. `clean_lat` returns `(lat, valid)` in Task 3, unpacked identically in Task 5.

**Known cross-task dependency.** Notebook variables flow across parts (`case`, `electric`, `stored_bipolar`, `points`, `triangles`, `lat`, `valid`, `surface_points`, `areas`, `nn_distance`, `raw`, `window_samples`, `vertex_lat`, `vertex_voltage`, `cv_speed`, `arrows`, `tags`). This is intentional for a linear teaching notebook; the clean-kernel run in Task 10 is what guarantees the chain holds.
