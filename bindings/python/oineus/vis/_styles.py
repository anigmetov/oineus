"""Default style dicts and tuning constants for matplotlib-backed plots.

These are mutable module-level defaults that every plotting helper consults.
Callers get a fresh dict from the ``default_*_style()`` getters, modify it,
and pass it back via the corresponding kwarg for a one-shot override. For
global overrides (once per script / notebook), mutate the module-level
``DEFAULT_*_STYLE`` dict in place.

The dict contents are matplotlib-flavoured (kwargs like ``marker``, ``cmap``,
``angles``). A plotly backend will keep its own analogues elsewhere.
"""
from __future__ import annotations


DEFAULT_POINT_STYLE: dict = {
    "marker": "o",
    "s": 25.0,
    "alpha": 0.9,
    "edgecolors": "none",
}

DEFAULT_DIAGRAM_A_POINT_STYLE: dict = {**DEFAULT_POINT_STYLE, "c": "tab:red"}
DEFAULT_DIAGRAM_B_POINT_STYLE: dict = {**DEFAULT_POINT_STYLE, "c": "tab:blue"}

DEFAULT_MATCHING_EDGE_STYLE: dict = {
    "linewidth": 0.8,
    "linestyle": "-",
    "color": "gray",
    "alpha": 0.5,
}

DEFAULT_LONGEST_EDGE_STYLE: dict = {
    "linewidth": 1.6,
    "linestyle": "-",
    "color": "red",
    "alpha": 0.95,
}

DEFAULT_DIAGONAL_STYLE: dict = {
    "linestyle": "--",
    "color": "gray",
    "alpha": 0.7,
    "linewidth": 1.0,
}

DEFAULT_DIAGONAL_PROJECTION_A_STYLE: dict = {
    "marker": "x",
    "s": 30.0,
    "alpha": 0.8,
    "c": "tab:red",
}

DEFAULT_DIAGONAL_PROJECTION_B_STYLE: dict = {
    "marker": "x",
    "s": 30.0,
    "alpha": 0.8,
    "c": "tab:blue",
}

DEFAULT_INF_LINE_STYLE: dict = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.0,
}

DEFAULT_DIAGRAM_GRADIENT_STYLE: dict = {
    "color": "tab:green",
    "alpha": 0.85,
    "width": 0.004,
    "headwidth": 3.5,
    "headlength": 5.0,
    "angles": "xy",
    "scale_units": "xy",
    "scale": 1.0,
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


# Chain rendering. Default to a single accent colour (orange) so the chain
# stands out against a desaturated grey point cloud / scalar-field heatmap.
DEFAULT_CHAIN_VERTEX_STYLE: dict = {
    "marker": "o",
    "s": 60.0,
    "c": "tab:orange",
    "edgecolors": "black",
    "linewidths": 0.8,
    "zorder": 3,
}

DEFAULT_CHAIN_EDGE_STYLE: dict = {
    "color": "tab:orange",
    "linewidth": 2.0,
    "alpha": 0.95,
    "zorder": 2,
}

DEFAULT_CHAIN_TRIANGLE_STYLE: dict = {
    "facecolor": "tab:orange",
    "edgecolor": "tab:orange",
    "alpha": 0.35,
    "linewidth": 0.8,
    "zorder": 1,
}

# Phase-2 (3D) -- defined here so callers can preview overrides.
DEFAULT_CHAIN_TETRAHEDRON_STYLE: dict = {
    "facecolor": "tab:orange",
    "edgecolor": "tab:orange",
    "alpha": 0.20,
    "linewidth": 0.6,
    "zorder": 1,
}

DEFAULT_POINT_CLOUD_STYLE: dict = {
    "marker": "o",
    "s": 12.0,
    "c": "lightgray",
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
