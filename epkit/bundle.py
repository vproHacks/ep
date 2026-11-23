"""The precomputed data bundle: build it once, read it anywhere.

The notebook used to load 67 MB of MATLAB with `openep-py`, `scipy.io`, and
PyVista. None of those three run in a browser, and the notebook is now published
as a WebAssembly page, so the heavy loading happens **once**, offline, in
`scripts/build_bundle.py`, and the result is a single compressed `.npz`.

What goes in is deliberately raw: waveforms, positions, timings, telemetry. The
analysis stays in the notebook, where a reader can see it. The only *derived*
values stored here are the ones that need `openep-py` itself, kept so that the
notebook can still check its own arithmetic against the reference implementation
(see `openep_lat_field`, `openep_voltage_field`, and the free boundaries).

Build:  python scripts/build_bundle.py
Load :  epkit.load_bundle(path_or_url)
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np

BUNDLE_NAME = "ep_primer_data.npz"

#: Waveform arrays are stored at reduced precision. The bipolar signal and the
#: ECG carry numbers the notebook quotes, so they keep float32; the unipolar
#: pair is only ever used for shape comparisons and correlations, so float16
#: (~3 significant digits) is enough and halves the download.
_PRECISION = {
    "bipolar_egm": np.float32,
    "ecg": np.float32,
    "unipolar_egm": np.float16,
}


class Bundle:
    """Attribute access over the arrays in the `.npz`.

    `bundle.points` beats `bundle["points"]` in a teaching notebook, and the
    class keeps the provenance metadata attached to the data it describes.
    """

    def __init__(self, arrays: dict):
        self._arrays = arrays
        self.meta = json.loads(str(arrays["meta_json"]))

    def __getattr__(self, name: str):
        try:
            value = self._arrays[name]
        except KeyError as exc:
            raise AttributeError(
                f"{name!r} is not in the bundle. Available: "
                f"{', '.join(sorted(self._arrays))}"
            ) from exc
        if isinstance(value, np.ndarray) and value.dtype == np.float16:
            return value.astype(np.float32)
        return value

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._arrays))

    def __repr__(self) -> str:
        megabytes = sum(
            a.nbytes for a in self._arrays.values() if isinstance(a, np.ndarray)
        ) / 1e6
        return (
            f"<Bundle {self.meta['source']} | {len(self._arrays)} arrays, "
            f"{megabytes:.1f} MB in memory>"
        )

    @property
    def boundary_rings(self) -> list[np.ndarray]:
        """Free-boundary vertex rings, one array of vertex indices per opening.

        Stored flat (npz holds rectangular arrays), reassembled here.
        """
        flat = self._arrays["boundary_ring_vertices"]
        offsets = self._arrays["boundary_ring_offsets"]
        return [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def load_bundle(location=None) -> Bundle:
    """Read a bundle from a local path or an http(s) URL.

    The URL branch is what makes the published notebook work: under Pyodide,
    `urllib` is patched to use the browser's fetch, so the same call that reads
    a file on a laptop reads an asset off GitHub Pages in the browser.
    """
    if location is None:
        location = Path(__file__).resolve().parent.parent / "public" / BUNDLE_NAME

    text = str(location)
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text) as response:  # noqa: S310 - fixed scheme
            payload = io.BytesIO(response.read())
    else:
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(
                f"No data bundle at {path}. Build it with:\n"
                f"    python scripts/build_bundle.py"
            )
        payload = path

    with np.load(payload, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    return Bundle(arrays)


# --------------------------------------------------------------------------
# Everything below runs offline, on a machine with openep-py installed.
# --------------------------------------------------------------------------


def _channel_range(values) -> tuple:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return (float("nan"), float("nan"))
    return (float(finite.min()), float(finite.max()))


def _ablation_ranges(case) -> dict:
    """Min/max of each RF telemetry channel — the audit the notebook repeats."""
    ablation = case.ablation
    return {
        "power": _channel_range(ablation.power),
        "temperature": _channel_range(ablation.temperature),
        "impedance": _channel_range(ablation.impedance),
        "force": _channel_range(ablation.force.force),
    }


def build_bundle(destination=None, dataset: int = 1) -> Path:
    """Read the OpenEP MATLAB exports and write the notebook's data bundle.

    Requires `openep-py`, `pyvista`, and numpy<2 — the build-time stack in
    `requirements-build.txt`. The notebook itself needs none of them.
    """
    import openep  # noqa: F401 - imported for the version string
    from openep.mesh.mesh_routines import get_free_boundaries

    from epkit.data import _dataset_path, load_case, load_rfindex

    import scipy.io

    if destination is None:
        destination = Path(__file__).resolve().parent.parent / "public" / BUNDLE_NAME
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    case = load_case(dataset)
    electric = case.electric
    annotations = electric.annotations
    tags = load_rfindex(dataset)
    raw = scipy.io.loadmat(str(_dataset_path(dataset)))
    userdata = raw["userdata"][0, 0]

    # Free boundaries need PyVista and OpenEP's own routine; resolve them here
    # so the notebook can compare its pure-topology answer against the library.
    mesh = case.create_mesh()
    free_boundaries = get_free_boundaries(mesh)
    separated = free_boundaries.separate_boundaries()
    ring_vertices = np.concatenate(
        [np.append(ring[:, 0], ring[0, 0]) for ring in separated]
    ).astype(np.int32)
    ring_offsets = np.concatenate(
        [[0], np.cumsum([len(ring) + 1 for ring in separated])]
    ).astype(np.int32)

    def _strings(field) -> np.ndarray:
        return np.array(
            [
                str(np.asarray(entry).ravel()[0]) if np.asarray(entry).size else ""
                for entry in np.asarray(field).ravel()
            ]
        )

    point_tags = _strings(userdata["electric"][0, 0]["tags"])
    electrode_names = _strings(userdata["electric"][0, 0]["electrodeNames_bip"])

    arrays: dict = {
        # ---- geometry -----------------------------------------------------
        "points": np.asarray(case.points, dtype=np.float32),
        "triangles": np.asarray(case.indices, dtype=np.int32),
        "boundary_ring_vertices": ring_vertices,
        "boundary_ring_offsets": ring_offsets,
        "boundary_lengths": np.asarray(
            free_boundaries.calculate_lengths(), dtype=np.float32
        ),
        "boundary_areas": np.asarray(
            free_boundaries.calculate_areas(), dtype=np.float32
        ),
        # ---- recordings ---------------------------------------------------
        "bipolar_egm": np.asarray(electric.bipolar_egm.egm),
        "unipolar_egm": np.asarray(electric.unipolar_egm.egm),
        "ecg": np.asarray(electric.ecg.ecg[:, :, 0]),
        "egm_points": np.asarray(electric.bipolar_egm.points, dtype=np.float32),
        "surface_points": np.asarray(electric.surface.nearest_point, dtype=np.float32),
        "electrode_positions": np.asarray(
            electric.unipolar_egm.points, dtype=np.float32
        ),
        "surface_normals": np.asarray(electric.surface.normals, dtype=np.float32),
        "stored_voltage": np.asarray(electric.bipolar_egm.voltage, dtype=np.float32),
        "stored_unipolar_voltage": np.asarray(
            electric.unipolar_egm.voltage, dtype=np.float32
        ),
        # ---- timing -------------------------------------------------------
        "local_activation_time": np.asarray(
            annotations.local_activation_time, dtype=np.float32
        ),
        "reference_activation_time": np.asarray(
            annotations.reference_activation_time, dtype=np.float32
        ),
        "window_of_interest": np.asarray(
            annotations.window_of_interest, dtype=np.float32
        ),
        # ---- labels -------------------------------------------------------
        "internal_names": np.asarray(electric.internal_names).astype("U16"),
        "electrode_names": electrode_names.astype("U16"),
        "point_tags": point_tags.astype("U32"),
        "ecg_channel_names": np.asarray(electric.ecg.channel_names).astype("U16"),
        # ---- what openep-py itself produced, for cross-checking ------------
        "openep_lat_field": np.asarray(
            case.fields.local_activation_time, dtype=np.float32
        ),
        "openep_voltage_field": np.asarray(
            case.fields.bipolar_voltage, dtype=np.float32
        ),
        # ---- therapy ------------------------------------------------------
        "ablation_times": np.asarray(case.ablation.times, dtype=np.float32),
        "ablation_power": np.asarray(case.ablation.power, dtype=np.float32),
        "ablation_temperature": np.asarray(case.ablation.temperature, dtype=np.float32),
        "ablation_impedance": np.asarray(case.ablation.impedance, dtype=np.float32),
        "force_times": np.asarray(case.ablation.force.times, dtype=np.float32),
        "force_values": np.asarray(case.ablation.force.force, dtype=np.float32),
        # ---- lesion tags (rfindex, which openep-py does not read) ----------
        "tag_points": np.asarray(tags.points, dtype=np.float32),
        "tag_duration": np.asarray(tags.duration, dtype=np.float32),
        "tag_avg_force": np.asarray(tags.avg_force, dtype=np.float32),
        "tag_max_temp": np.asarray(tags.max_temp, dtype=np.float32),
        "tag_max_power": np.asarray(tags.max_power, dtype=np.float32),
        "tag_fti": np.asarray(tags.fti, dtype=np.float32),
        "tag_index_value": np.asarray(tags.index_value, dtype=np.float32),
    }

    for key, dtype in _PRECISION.items():
        arrays[key] = arrays[key].astype(dtype)

    other = 2 if dataset == 1 else 1
    meta = {
        "source": f"OpenEP MATLAB export, dataset {dataset}",
        "provenance": (
            "Built by scripts/build_bundle.py from the .mat files distributed "
            "with openep-py. Waveforms are stored at reduced precision "
            "(bipolar/ECG float32, unipolar float16); everything else is "
            "float32 or int32."
        ),
        "sampling_frequency_hz": float(electric.frequency),
        "n_samples": int(electric.n_samples),
        "n_points": int(electric.n_points),
        "openep_version": getattr(openep, "__version__", "unknown"),
        "dataset": dataset,
        "reference_dataset": other,
        "ablation_ranges": _ablation_ranges(case),
        "ablation_ranges_other_dataset": _ablation_ranges(load_case(other)),
        "index_name": tags.index_name,
    }
    arrays["meta_json"] = np.array(json.dumps(meta, indent=1))

    np.savez_compressed(destination, **arrays)
    return destination
