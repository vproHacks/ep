"""The data bundle is the notebook's only input, so it carries the most tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from epkit.bundle import BUNDLE_NAME, Bundle, load_bundle

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "public" / BUNDLE_NAME


def _write_toy_bundle(path):
    meta = {"source": "toy", "n_points": 2, "sampling_frequency_hz": 1000.0}
    np.savez_compressed(
        path,
        points=np.zeros((4, 3), dtype=np.float32),
        waveform=np.ones((2, 3), dtype=np.float16),
        names=np.array(["P1", "P2"]).astype("U16"),
        boundary_ring_vertices=np.array([0, 1, 2, 0, 2, 3, 2], dtype=np.int32),
        boundary_ring_offsets=np.array([0, 4, 7], dtype=np.int32),
        meta_json=np.array(json.dumps(meta)),
    )
    return path


def test_round_trip_attribute_access(tmp_path):
    bundle = load_bundle(_write_toy_bundle(tmp_path / "toy.npz"))
    assert isinstance(bundle, Bundle)
    assert bundle.points.shape == (4, 3)
    assert list(bundle.names) == ["P1", "P2"]
    assert bundle.meta["source"] == "toy"


def test_float16_arrays_are_upcast_on_read(tmp_path):
    """float16 halves the download; arithmetic on it overflows and is slow."""
    bundle = load_bundle(_write_toy_bundle(tmp_path / "toy.npz"))
    assert bundle.waveform.dtype == np.float32


def test_boundary_rings_are_reassembled(tmp_path):
    bundle = load_bundle(_write_toy_bundle(tmp_path / "toy.npz"))
    rings = bundle.boundary_rings
    assert len(rings) == 2
    assert rings[0].tolist() == [0, 1, 2, 0]
    assert rings[1].tolist() == [2, 3, 2]


def test_unknown_key_names_what_is_available(tmp_path):
    bundle = load_bundle(_write_toy_bundle(tmp_path / "toy.npz"))
    with pytest.raises(AttributeError, match="points"):
        bundle.no_such_array


def test_missing_bundle_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_bundle"):
        load_bundle(tmp_path / "absent.npz")


def test_loads_over_http(tmp_path):
    """The published notebook reads the bundle by URL, so that path is tested.

    In the browser this same call goes through pyodide's patched urllib and out
    to GitHub Pages; here it goes to a throwaway local server.
    """
    import functools
    import http.server
    import threading

    _write_toy_bundle(tmp_path / "toy.npz")
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        bundle = load_bundle(f"http://127.0.0.1:{server.server_port}/toy.npz")
        assert bundle.points.shape == (4, 3)
        assert bundle.meta["source"] == "toy"
    finally:
        server.shutdown()


# --------------------------------------------------------------------------
# The real bundle, when it has been built. These are the teaching facts the
# notebook narrates, pinned so a rebuild cannot silently change the story.
# --------------------------------------------------------------------------

real_bundle = pytest.mark.skipif(
    not BUNDLE_PATH.exists(), reason="run scripts/build_bundle.py first"
)


@pytest.fixture(scope="module")
def built():
    return load_bundle(BUNDLE_PATH)


@real_bundle
def test_geometry_shape(built):
    assert built.points.shape == (9279, 3)
    assert built.triangles.shape == (10296, 3)


@real_bundle
def test_recording_shapes(built):
    assert built.bipolar_egm.shape == (1366, 2500)
    assert built.unipolar_egm.shape == (1366, 2500, 2)
    assert built.ecg.shape == (1366, 2500)
    assert built.meta["sampling_frequency_hz"] == 1000.0


@real_bundle
def test_ecg_is_single_channel(built):
    """Real PASO needs 12 leads. This export has one. Part 9 says so."""
    assert list(built.ecg_channel_names) == ["ECG"]


@real_bundle
def test_lat_sentinels_survive_the_repack(built):
    lat = built.local_activation_time - built.reference_activation_time
    assert int(np.sum(lat <= -1000)) == 22, "the -10000 sentinel count is a teaching fact"


@real_bundle
def test_six_free_boundaries_one_of_them_much_the_largest(built):
    rings = built.boundary_rings
    assert len(rings) == 6, "mitral valve, four pulmonary veins, appendage"
    lengths = np.sort(built.boundary_lengths)[::-1]
    assert lengths[0] > 1.4 * lengths[1], "the mitral valve dominates"


@real_bundle
def test_ninety_lesion_tags_with_physiological_metrics(built):
    assert built.tag_points.shape == (90, 3)
    assert 0 < np.nanmedian(built.tag_avg_force) < 60, "grams-force"
    assert 20 < np.nanmedian(built.tag_max_power) < 60, "watts"


@real_bundle
def test_rf_telemetry_is_unpopulated_as_the_notebook_claims(built):
    assert np.nanmax(built.ablation_power) == 0.0
    assert np.nanmax(built.ablation_temperature) < 40.0
    impedance = built.ablation_impedance.astype(float)
    assert np.allclose(np.diff(impedance[:6]), 100.0), "a counter, not ohms"


@real_bundle
def test_point_tags_are_present_and_empty(built):
    assert len(built.point_tags) == 1366
    assert [t for t in built.point_tags if t.strip() not in ("", "[]")] == []


@real_bundle
def test_bipolar_precision_survives_the_repack(built):
    """Part 6 claims peak-to-peak reproduces the system's voltage exactly.

    That claim is only true if the bipolar waveforms kept enough precision
    through the float32 repack, so it doubles as the fidelity test.
    """
    bipolar = built.bipolar_egm.astype(float)
    reference = built.reference_activation_time
    woi = built.window_of_interest

    ours = np.full(len(bipolar), np.nan)
    for i in range(len(bipolar)):
        start = max(int(reference[i] + woi[i, 0]), 0)
        stop = min(int(reference[i] + woi[i, 1]), bipolar.shape[1])
        if stop - start < 2:
            continue
        ours[i] = np.ptp(bipolar[i, start:stop])

    stored = built.stored_voltage.astype(float)
    comparable = np.isfinite(ours) & np.isfinite(stored)
    assert np.nanmax(np.abs(ours[comparable] - stored[comparable])) < 1e-3
