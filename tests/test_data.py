"""The offline loaders: MATLAB in, arrays out.

Skipped unless the build-time stack is installed and the datasets are cached,
because none of it is needed to run the notebook — see requirements-build.txt.
"""

import numpy as np
import pytest

pytest.importorskip("openep", reason="build-time only; see requirements-build.txt")

from epkit.data import DATA_DIR, load_case, load_rfindex  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "openep_dataset_1.mat").exists(),
    reason="datasets not cached; run scripts/build_bundle.py",
)


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
