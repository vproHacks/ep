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

    matrix = np.stack([e1, e2, normal], axis=1)  # (n_tri, 3, 3)
    rhs = np.stack([s1 - s0, s2 - s0, np.zeros(len(triangles))], axis=1)

    grad = np.full((len(triangles), 3), np.nan)
    ok = np.isfinite(rhs).all(axis=1) & np.isfinite(matrix).all(axis=(1, 2))
    if np.any(ok):
        determinant = np.zeros(len(triangles))
        determinant[ok] = np.linalg.det(matrix[ok])
        ok &= np.abs(determinant) > 1e-12
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
