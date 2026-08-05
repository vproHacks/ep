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
from epkit.bundle import BUNDLE_NAME, Bundle, load_bundle
from epkit.render import (
    LAT_CMAP,
    VOLTAGE_CMAP,
    mesh_to_plotly,
    plotly_figure,
    voltage_colorscale,
)

__all__ = [
    "BUNDLE_NAME",
    "Bundle",
    "LAT_CMAP",
    "VOLTAGE_CMAP",
    "clean_lat",
    "conduction_velocity",
    "fractionation_index",
    "load_bundle",
    "mesh_to_plotly",
    "plotly_figure",
    "surface_gradient",
    "voltage_colorscale",
]
