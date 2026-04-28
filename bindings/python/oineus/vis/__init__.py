"""Persistence-diagram and chain visualization helpers.

The matplotlib functions live in :mod:`._matplotlib`; the plotly mirror
will land in :mod:`oineus.plotly`. Style-dict defaults and tuning
constants live in :mod:`._styles`. Backend-agnostic helpers are in
:mod:`._common`.

Public symbols are re-exported here and from the package root
(``oineus.plot_diagram``, ``oineus.DEFAULT_DENSITY_STYLE``, ...).
"""
from __future__ import annotations

from ._matplotlib import (
    plot_chain,
    plot_diagram,
    plot_diagram_gradient,
    plot_matching,
    _add_density_artist,
    _require_scatter_density,
)
from ._styles import (
    DEFAULT_CHAIN_EDGE_STYLE,
    DEFAULT_CHAIN_TETRAHEDRON_STYLE,
    DEFAULT_CHAIN_TRIANGLE_STYLE,
    DEFAULT_CHAIN_VERTEX_STYLE,
    DEFAULT_DENSITY_STYLE,
    DEFAULT_DENSITY_THRESHOLD,
    DEFAULT_DIAGONAL_PROJECTION_A_STYLE,
    DEFAULT_DIAGONAL_PROJECTION_B_STYLE,
    DEFAULT_DIAGONAL_STYLE,
    DEFAULT_DIAGRAM_A_POINT_STYLE,
    DEFAULT_DIAGRAM_B_POINT_STYLE,
    DEFAULT_DIAGRAM_GRADIENT_STYLE,
    DEFAULT_GRADIENT_TOP_K_ARROWS,
    DEFAULT_INF_LINE_STYLE,
    DEFAULT_LONGEST_EDGE_STYLE,
    DEFAULT_MATCHING_EDGE_QUANTILE,
    DEFAULT_MATCHING_EDGE_STYLE,
    DEFAULT_POINT_CLOUD_STYLE,
    DEFAULT_POINT_STYLE,
    default_chain_edge_style,
    default_chain_tetrahedron_style,
    default_chain_triangle_style,
    default_chain_vertex_style,
    default_density_style,
    default_diagonal_projection_a_style,
    default_diagonal_projection_b_style,
    default_diagonal_style,
    default_diagram_a_point_style,
    default_diagram_b_point_style,
    default_diagram_gradient_style,
    default_inf_line_style,
    default_longest_edge_style,
    default_matching_edge_style,
    default_point_cloud_style,
    default_point_style,
)


__all__ = [
    "plot_chain",
    "plot_diagram",
    "plot_diagram_gradient",
    "plot_matching",
    # Constants
    "DEFAULT_CHAIN_EDGE_STYLE",
    "DEFAULT_CHAIN_TETRAHEDRON_STYLE",
    "DEFAULT_CHAIN_TRIANGLE_STYLE",
    "DEFAULT_CHAIN_VERTEX_STYLE",
    "DEFAULT_DENSITY_STYLE",
    "DEFAULT_DENSITY_THRESHOLD",
    "DEFAULT_DIAGONAL_PROJECTION_A_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_B_STYLE",
    "DEFAULT_DIAGONAL_STYLE",
    "DEFAULT_DIAGRAM_A_POINT_STYLE",
    "DEFAULT_DIAGRAM_B_POINT_STYLE",
    "DEFAULT_DIAGRAM_GRADIENT_STYLE",
    "DEFAULT_GRADIENT_TOP_K_ARROWS",
    "DEFAULT_INF_LINE_STYLE",
    "DEFAULT_LONGEST_EDGE_STYLE",
    "DEFAULT_MATCHING_EDGE_QUANTILE",
    "DEFAULT_MATCHING_EDGE_STYLE",
    "DEFAULT_POINT_CLOUD_STYLE",
    "DEFAULT_POINT_STYLE",
    # Getters
    "default_chain_edge_style",
    "default_chain_tetrahedron_style",
    "default_chain_triangle_style",
    "default_chain_vertex_style",
    "default_density_style",
    "default_diagonal_projection_a_style",
    "default_diagonal_projection_b_style",
    "default_diagonal_style",
    "default_diagram_a_point_style",
    "default_diagram_b_point_style",
    "default_diagram_gradient_style",
    "default_inf_line_style",
    "default_longest_edge_style",
    "default_matching_edge_style",
    "default_point_cloud_style",
    "default_point_style",
]
