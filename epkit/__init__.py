"""Support code for the EP primer notebook.

Only loading, rendering, and tested analytical helpers live here. The
explanatory mathematics stays visible in the notebook cells.
"""

from epkit.analysis import (
    clean_lat,
    conduction_velocity,
    fractionation_index,
    surface_gradient,
)
from epkit.data import DATA_DIR, LesionTags, load_case, load_rfindex
from epkit.render import (
    LAT_CMAP,
    VOLTAGE_CMAP,
    decimate,
    mesh_to_plotly,
    plotly_figure,
    show_pyvista,
    voltage_colorscale,
)

__all__ = [
    "DATA_DIR",
    "LAT_CMAP",
    "LesionTags",
    "VOLTAGE_CMAP",
    "clean_lat",
    "conduction_velocity",
    "decimate",
    "fractionation_index",
    "load_case",
    "load_rfindex",
    "mesh_to_plotly",
    "plotly_figure",
    "show_pyvista",
    "surface_gradient",
    "voltage_colorscale",
]
