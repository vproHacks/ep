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

## Layout

| Path | Contents |
|---|---|
| `ep_primer.ipynb` | The notebook |
| `scripts/build_notebook.py` | Generates the notebook, so it is reviewable in git |
| `epkit/data.py` | Dataset download/cache, case loader, `rfindex` lesion-tag parser |
| `epkit/analysis.py` | Conduction velocity, LAT cleaning, fractionation |
| `epkit/render.py` | Plotly/PyVista helpers and clinical colour scales |
| `docs/superpowers/` | Design spec and implementation plan |
