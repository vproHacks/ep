import numpy as np


def test_numpy_is_pinned_below_2():
    assert np.__version__.startswith("1."), (
        f"openep requires numpy<2 (np.NaN); got {np.__version__}"
    )


def test_openep_imports():
    import openep

    assert hasattr(openep, "load_openep_mat")
