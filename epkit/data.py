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
    """Parse the ablation lesion tags out of the rfindex struct."""
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
