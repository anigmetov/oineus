import numpy as np
import eagerpy as epy

from .. import _oineus
from .diff_filtration import DiffFiltration
from ._tensor_utils import tensor_to_real_numpy, gather_values


_FR_GRID_CLASS_BY_NDIM = {1: _oineus.Grid_1D, 2: _oineus.Grid_2D, 3: _oineus.Grid_3D}


def freudenthal_filtration(data, negate=False, wrap=False, max_dim=3, slim=False, n_threads=1):
    tensor = epy.astensor(data)
    np_data = tensor_to_real_numpy(tensor)
    max_dim = min(max_dim, np_data.ndim)
    # slim=True builds the compact (anchor,type) FreudenthalFiltration_ND for D=1,2,3 on non-wrap
    # grids: it reduces, produces diagrams and optimizes (sorted_id + value based) identically to
    # the fat path but with a far smaller boundary-build footprint, and oineus.diff.TopologyOptimizer
    # dispatches on its per-dim type. It is opt-in for the differentiable factory (the non-diff
    # oineus.freudenthal_filtration defaults to slim) because some low-level diff entry points
    # (compute_partial_u_rows, the raw per-cell optimizer ctor) are still fat-only; the high-level
    # diff pipeline (persistence_diagram, TopologyOptimizer, mapping_cylinder) handles slim. wrap
    # grids and D>=4 always use the fat universal Filtration (FrGeometry rejects wrap; the slim
    # builder is bound only for D=1,2,3). The critical-vertex array cv is a flat index array into
    # the data, identical between the slim and fat paths, so the gathered values are unchanged.
    if slim and (not wrap) and (1 <= np_data.ndim <= 3):
        grid = _FR_GRID_CLASS_BY_NDIM[np_data.ndim](np_data, wrap=wrap, values_on="vertices")
        fil, cv = grid.freudenthal_filtration_and_critical_vertices_slim(
            max_dim=max_dim, negate=negate, n_threads=n_threads
        )
    else:
        fil, cv = _oineus.get_freudenthal_filtration_and_crit_vertices(
            np_data, negate, wrap, max_dim, n_threads
        )
    values = gather_values(tensor, cv)
    return DiffFiltration(fil, values)
