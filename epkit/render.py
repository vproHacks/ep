"""Rendering boilerplate: Plotly meshes and the clinical colour scales.

Plotly is the only 3D engine left in the stack. It renders in the browser with
no local install, which is what lets the published notebook show the same
rotatable heart a laptop does.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Early activation red through to late blue, the convention on clinical systems.
LAT_CMAP = "turbo_r"
VOLTAGE_CMAP = "jet_r"

_BACKGROUND = "rgba(0,0,0,0)"


def voltage_colorscale(low=0.5, high=1.5, vmax=5.0):
    """Clinical bipolar-voltage scale.

    Below `low` mV is dense scar (red), above `high` mV is healthy myocardium
    (purple/blue), between them is border zone. Breakpoints are placed at the
    real thresholds rather than spread evenly.
    """
    lo, hi = low / vmax, high / vmax
    return [
        [0.0, "rgb(190,30,30)"],
        [max(lo - 1e-6, 0.0), "rgb(230,90,50)"],
        [lo, "rgb(240,200,60)"],
        [(lo + hi) / 2, "rgb(70,190,90)"],
        [hi, "rgb(60,140,220)"],
        [1.0, "rgb(120,60,180)"],
    ]


def mesh_to_plotly(
    points,
    triangles,
    scalars=None,
    *,
    colorscale="Turbo",
    cmin=None,
    cmax=None,
    name="",
    opacity=1.0,
    colorbar_title="",
    showscale=True,
):
    """Wrap a triangulated surface as a Plotly Mesh3d trace."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)

    kwargs = dict(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name=name,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
    )

    if scalars is None:
        kwargs.update(color="rgb(200,180,175)")
    else:
        scalars = np.asarray(scalars, dtype=float)
        finite = scalars[np.isfinite(scalars)]
        kwargs.update(
            intensity=scalars,
            colorscale=colorscale,
            cmin=cmin if cmin is not None else (finite.min() if finite.size else 0.0),
            cmax=cmax if cmax is not None else (finite.max() if finite.size else 1.0),
            showscale=showscale,
            colorbar=dict(title=colorbar_title, thickness=14, len=0.7),
        )
    return go.Mesh3d(**kwargs)


def plotly_figure(traces, title="", height=650):
    """Assemble traces into an equal-aspect 3D figure."""
    fig = go.Figure(data=list(traces))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        paper_bgcolor=_BACKGROUND,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=_BACKGROUND,
        ),
    )
    return fig
