"""Default style dicts and tuning constants for matplotlib-backed plots.

These are mutable module-level defaults that every plotting helper consults.
Callers get a fresh dict from the ``default_*_style()`` getters, modify it,
and pass it back via the corresponding kwarg for a one-shot override. For
global overrides (once per script / notebook), mutate the module-level
``DEFAULT_*_STYLE`` dict in place.

The dict contents are matplotlib-flavoured (kwargs like ``marker``, ``cmap``,
``angles``). A plotly backend will keep its own analogues elsewhere.

Defaults track the Okabe-Ito (Wong 2011) colorblind-safe palette. The
recommended single-diagram look (preset P11b in the eval gallery) is a
saturated blue with a darker-shade edge halo: marker fill darker than
plain matplotlib defaults, edge a darkened version of the same hue so
clusters of overlapping points don't develop the white-halo artefact
that an "edgecolors='white'" default produces on a white background.
"""
from __future__ import annotations

from matplotlib.colors import to_rgb


# ---------------------------------------------------------------------------
# Okabe-Ito palette (Wong, Nature Methods 2011)
# ---------------------------------------------------------------------------

OKABE_ITO_BLUE: str = "#0072B2"
OKABE_ITO_VERMILLION: str = "#D55E00"
OKABE_ITO_GREEN: str = "#009E73"
OKABE_ITO_ORANGE: str = "#E69F00"
OKABE_ITO_SKY: str = "#56B4E9"
OKABE_ITO_PURPLE: str = "#CC79A7"
OKABE_ITO_YELLOW: str = "#F0E442"
OKABE_ITO_BLACK: str = "#000000"


def _darken(color, amount: float = 0.55) -> tuple:
    """Multiply RGB channels by ``amount`` to get a darker shade.

    Used to build edge colours that match the marker fill but read as
    a halo darker than the fill, so dense clusters fill in solidly and
    isolated outliers still get a visible boundary.
    """
    r, g, b = to_rgb(color)
    return (r * amount, g * amount, b * amount)


_DARK_BLUE = _darken(OKABE_ITO_BLUE)
_DARK_VERMILLION = _darken(OKABE_ITO_VERMILLION)


DEFAULT_POINT_STYLE: dict = {
    "marker": "o",
    "s": 8.0,
    "alpha": 0.85,
    "edgecolors": _DARK_BLUE,
    "linewidths": 0.45,
}

# Color lives outside the style dicts: callers pass it explicitly via
# the top-level color / color_dgm_a / color_dgm_b arguments on
# plot_diagram / plot_matching. The defaults below remain matplotlib
# kwargs (marker, size, alpha, edge styling) only.
DEFAULT_DIAGRAM_A_POINT_STYLE: dict = dict(DEFAULT_POINT_STYLE)
DEFAULT_DIAGRAM_B_POINT_STYLE: dict = dict(
    DEFAULT_POINT_STYLE, edgecolors=_DARK_VERMILLION,
)

# Default colors when the user does not pass color / color_dgm_a / color_dgm_b.
DEFAULT_DIAGRAM_A_COLOR: str = OKABE_ITO_BLUE
DEFAULT_DIAGRAM_B_COLOR: str = OKABE_ITO_VERMILLION
DEFAULT_DIAGRAM_GRADIENT_DIAGRAM_COLOR: str = OKABE_ITO_BLUE
DEFAULT_DIAGRAM_GRADIENT_GRAD_COLOR: str = OKABE_ITO_GREEN
DEFAULT_MATCHING_EDGE_COLOR: str = "0.30"
DEFAULT_CHAIN_COLOR: str = OKABE_ITO_ORANGE
DEFAULT_POINT_CLOUD_COLOR: str = "lightgray"

DEFAULT_MATCHING_EDGE_STYLE: dict = {
    "linewidth": 1.0,
    "linestyle": "-",
    "alpha": 0.65,
}

DEFAULT_LONGEST_EDGE_STYLE: dict = {
    "linewidth": 1.6,
    "linestyle": "-",
    "color": "red",
    "alpha": 0.95,
}

DEFAULT_DIAGONAL_STYLE: dict = {
    "linestyle": "-",
    "color": "0.55",
    "alpha": 0.85,
    "linewidth": 0.7,
}

# Color is filled in by plot_matching from color_dgm_a / color_dgm_b.
DEFAULT_DIAGONAL_PROJECTION_A_STYLE: dict = {
    "marker": "x",
    "s": 30.0,
    "alpha": 0.8,
}

DEFAULT_DIAGONAL_PROJECTION_B_STYLE: dict = {
    "marker": "x",
    "s": 30.0,
    "alpha": 0.8,
}

DEFAULT_INF_LINE_STYLE: dict = {
    "color": "0.30",
    "linestyle": "-",
    "linewidth": 0.7,
}

# Subtle dotted grid drawn under the data. Enabled by default in
# plot_diagram so users get coordinate cues without having to set
# rcParams. Pass ``grid_style=False`` to disable, or override the dict.
DEFAULT_GRID_STYLE: dict = {
    "linestyle": ":",
    "linewidth": 0.5,
    "color": "0.85",
    "alpha": 1.0,
    "zorder": 0,
}

# Style for the markers that sit at the inf-line (death = +-inf). Distinct
# from DEFAULT_POINT_STYLE so the marker shape can carry the "off the chart"
# meaning -- upward triangle for +inf, conventional in TDA papers.
DEFAULT_INF_POINT_STYLE: dict = {
    "marker": "^",
    "s": 36.0,
    "alpha": 0.95,
    "edgecolors": _DARK_BLUE,
    "linewidths": 0.5,
}

DEFAULT_DIAGRAM_GRADIENT_STYLE: dict = {
    "alpha": 0.95,
    "width": 0.0045,
    "headwidth": 3.5,
    "headlength": 5.0,
    "angles": "xy",
    "scale_units": "xy",
    # In data coordinates, scale > 1 makes arrows shorter than the raw
    # gradient by that factor. Default scale=4.0 keeps arrows from
    # leaving the diagram bounding box on noisy inputs; pass scale=1.0
    # in quiver_style to recover raw-gradient lengths.
    "scale": 4.0,
}

# Above this point count we switch large diagrams from per-point scatter to
# a density-aggregated raster (via mpl_scatter_density). Tuned so that the
# perceptual oversaturation that starts around 5k is mostly hidden, while
# small diagrams still get the crisp scatter rendering users expect.
DEFAULT_DENSITY_THRESHOLD: int = 20_000

# Style passed to the underlying ScatterDensityArtist. The norm is set by
# the caller (defaults to PowerNorm(0.5) inside _add_density_artist).
DEFAULT_DENSITY_STYLE: dict = {
    "cmap": "viridis",
    "dpi": 72,
    "downres_factor": 4,
}

# Edges with length below this quantile are hidden in matching-density mode.
# 0.99 keeps the top 1% of edges -- typically the ones that carry the
# matching-cost signal.
DEFAULT_MATCHING_EDGE_QUANTILE: float = 0.99

# Number of arrows to draw in gradient-density mode (largest |grad| points).
DEFAULT_GRADIENT_TOP_K_ARROWS: int = 200


# Chain rendering. Color lives outside these dicts (top-level
# chain_color / point_cloud_color args on plot_chain).
DEFAULT_CHAIN_VERTEX_STYLE: dict = {
    "marker": "o",
    "s": 60.0,
    "edgecolors": "black",
    "linewidths": 0.8,
    "zorder": 3,
}

DEFAULT_CHAIN_EDGE_STYLE: dict = {
    "linewidth": 2.0,
    "alpha": 0.95,
    "zorder": 2,
}

DEFAULT_CHAIN_TRIANGLE_STYLE: dict = {
    "alpha": 0.35,
    "linewidth": 0.8,
    "zorder": 1,
}

# 3D chain style -- defined here so callers can preview overrides.
DEFAULT_CHAIN_TETRAHEDRON_STYLE: dict = {
    "alpha": 0.20,
    "linewidth": 0.6,
    "zorder": 1,
}

DEFAULT_POINT_CLOUD_STYLE: dict = {
    "marker": "o",
    "s": 12.0,
    "alpha": 0.7,
    "edgecolors": "none",
    "zorder": 0,
}


def default_point_style() -> dict:
    return dict(DEFAULT_POINT_STYLE)


def default_diagram_a_point_style() -> dict:
    return dict(DEFAULT_DIAGRAM_A_POINT_STYLE)


def default_diagram_b_point_style() -> dict:
    return dict(DEFAULT_DIAGRAM_B_POINT_STYLE)


def default_matching_edge_style() -> dict:
    return dict(DEFAULT_MATCHING_EDGE_STYLE)


def default_longest_edge_style() -> dict:
    return dict(DEFAULT_LONGEST_EDGE_STYLE)


def default_diagonal_style() -> dict:
    return dict(DEFAULT_DIAGONAL_STYLE)


def default_diagonal_projection_a_style() -> dict:
    return dict(DEFAULT_DIAGONAL_PROJECTION_A_STYLE)


def default_diagonal_projection_b_style() -> dict:
    return dict(DEFAULT_DIAGONAL_PROJECTION_B_STYLE)


def default_inf_line_style() -> dict:
    return dict(DEFAULT_INF_LINE_STYLE)


def default_inf_point_style() -> dict:
    return dict(DEFAULT_INF_POINT_STYLE)


def default_grid_style() -> dict:
    return dict(DEFAULT_GRID_STYLE)


def default_diagram_gradient_style() -> dict:
    return dict(DEFAULT_DIAGRAM_GRADIENT_STYLE)


def default_density_style() -> dict:
    return dict(DEFAULT_DENSITY_STYLE)


def default_chain_vertex_style() -> dict:
    return dict(DEFAULT_CHAIN_VERTEX_STYLE)


def default_chain_edge_style() -> dict:
    return dict(DEFAULT_CHAIN_EDGE_STYLE)


def default_chain_triangle_style() -> dict:
    return dict(DEFAULT_CHAIN_TRIANGLE_STYLE)


def default_chain_tetrahedron_style() -> dict:
    return dict(DEFAULT_CHAIN_TETRAHEDRON_STYLE)


def default_point_cloud_style() -> dict:
    return dict(DEFAULT_POINT_CLOUD_STYLE)
