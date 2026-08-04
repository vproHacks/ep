import numpy as np

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
