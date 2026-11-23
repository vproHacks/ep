# EP Primer

An interactive introduction to cardiac electrophysiology data — what a catheter
actually measures, how the coloured 3D heart maps are made, and how much of those
maps is measurement rather than inference.

Built on real clinical [OpenEP](https://openep.io) recordings from a catheter
ablation. **No medical background assumed**: every clinical term is explained
where it first appears, and there is a searchable glossary at the end.

It is a [marimo](https://marimo.io) notebook, so it runs three ways: locally as
an editable notebook, locally as an app, and — with no install at all — as a
WebAssembly page on GitHub Pages.

## Run it

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/marimo edit ep_primer.py     # editable notebook
.venv/bin/marimo run ep_primer.py      # read-only app
```

No dataset download, no MATLAB, no VTK: the notebook reads
`public/ep_primer_data.npz`, which is in the repository.

## Publish it

```bash
.venv/bin/marimo export html-wasm ep_primer.py -o _site --mode run --show-code
python -m http.server --directory _site
```

`.github/workflows/pages.yml` does exactly this on every push to `master` and
deploys the result to GitHub Pages. The reader's browser runs the Python.

## Rebuild the data bundle

Only needed if you want to regenerate `public/ep_primer_data.npz` from the
original MATLAB exports. This is the one part of the project that needs the
awkward pinned stack, which is precisely why it happens offline and once.

```bash
uv venv .venv-build --python 3.10
uv pip install --python .venv-build/bin/python "setuptools<81" wheel
uv pip install --python .venv-build/bin/python --no-build-isolation -r requirements-build.txt
.venv-build/bin/python scripts/build_bundle.py
```

The pins are mandatory: `openep-py` calls `np.NaN`, which NumPy 2 removed, and
imports `pkg_resources` at build time. The script downloads and caches ~67 MB of
`.mat` files into `data/` on first run.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The tests that need `openep-py` and the MATLAB files skip themselves when either
is absent, so the suite runs on the light stack and in CI.

## Layout

| Path | Contents |
|---|---|
| `ep_primer.py` | The notebook. A plain Python file — reviewable in git, executable by marimo |
| `public/ep_primer_data.npz` | The precomputed data bundle the notebook reads (12 MB) |
| `epkit/bundle.py` | Builds the bundle offline; loads it from a path or a URL |
| `epkit/analysis.py` | Conduction velocity, LAT cleaning, fractionation |
| `epkit/render.py` | Plotly mesh helpers and the clinical colour scales |
| `epkit/data.py` | MATLAB loading and the `rfindex` lesion-tag parser (offline only) |
| `scripts/build_bundle.py` | Regenerates the bundle from the MATLAB exports |
| `.github/workflows/pages.yml` | Tests, WASM export, Pages deploy |
| `docs/superpowers/` | Design spec and implementation plan |

## Why a data bundle

`openep-py` needs `numpy<2`, MATLAB struct parsing and PyVista. None of that
exists in a browser, and the two `.mat` files are 67 MB. So the heavy loading
runs once, offline, and writes a single compressed archive of plain arrays:

```
data/openep_dataset_1.mat  --(openep-py, scipy.io, pyvista)-->  public/ep_primer_data.npz
                                                                     |
                                              (numpy + scipy only)   v
                                                        laptop, or browser
```

The repacking unwraps structs and reduces waveform precision. It does no
filtering, resampling or cleaning: every derived quantity is computed in the
notebook's own cells, in front of the reader.
