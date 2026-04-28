"""Backward-compat shim. Everything moved to :mod:`oineus.vis`.

Existing code that does ``from oineus.vis_utils import plot_diagram`` keeps
working; tests that introspect the matplotlib-availability flags via
``oineus.vis_utils._HAS_MPL_SCATTER_DENSITY`` keep working too.
"""
from __future__ import annotations

from .vis import *  # noqa: F401,F403
from .vis._matplotlib import (  # noqa: F401
    _HAS_MATPLOTLIB,
    _HAS_MPL_SCATTER_DENSITY,
    _add_density_artist,
    _require_scatter_density,
)
from .vis._common import (  # noqa: F401
    _array_diagram,
    _build_diagram_arrays,
    _classify_points,
    _coerce_chain,
    _coerce_diagram_with_grad,
    _compute_plot_limits,
    _diagram_points_to_array,
    _point_coords_for_edge,
    _resolve_color,
    _resolve_style,
    _shift_for_log,
    _split_near_diagonal,
    _to_dim_diagrams,
)
