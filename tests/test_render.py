import numpy as np
import plotly.graph_objects as go

from epkit.render import mesh_to_plotly, plotly_figure, voltage_colorscale


def _tetra():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
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
