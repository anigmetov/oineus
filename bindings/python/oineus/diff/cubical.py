from typing import Optional

import numpy as np
import eagerpy as epy

from .. import _oineus
from .diff_filtration import DiffFiltration
from ._tensor_utils import tensor_to_real_numpy, gather_values


_GRID_CLASS_BY_NDIM = {
    1: _oineus.Grid_1D,
    2: _oineus.Grid_2D,
    3: _oineus.Grid_3D,
}


def cube_filtration(data,
                    negate: bool = False,
                    wrap: bool = False,
                    max_dim: Optional[int] = None,
                    values_on: str = "vertices",
                    n_threads: int = 1) -> DiffFiltration:
    """Differentiable cubical filtration from tensor-valued grid data.

    Mirrors `oineus.cube_filtration`, but returns a `DiffFiltration` whose
    filtration values are gathered back onto the autograd-tracked input.

    Args:
        data: 1D/2D/3D tensor-like (PyTorch/JAX/NumPy) of grid values.
        negate: Compute upper-star (superlevel) instead of lower-star.
        wrap: Not supported; raises if True.
        max_dim: Maximal cube dimension; defaults to ``data.ndim``.
        values_on: ``"vertices"`` (default) or ``"cells"``. Critical indices
            are flat indices into the original data array in either case.
        n_threads: Threads for C++ filtration construction.
    """
    if wrap:
        raise RuntimeError("cube_filtration: wrap=True is not supported")

    tensor = epy.astensor(data)
    np_data = tensor_to_real_numpy(tensor)
    ndim = np_data.ndim

    grid_cls = _GRID_CLASS_BY_NDIM.get(ndim)
    if grid_cls is None:
        raise RuntimeError(
            f"cube_filtration: data.ndim={ndim} not supported, must be 1, 2, or 3"
        )

    if max_dim is None:
        max_dim = ndim

    grid = grid_cls(np_data, wrap=wrap, values_on=values_on)
    fil, crit = grid.cube_filtration_and_critical_indices(
        max_dim=max_dim, negate=negate, n_threads=n_threads,
    )

    crit = np.asarray(crit, dtype=np.int64)
    values = gather_values(tensor, crit)
    return DiffFiltration(fil, values)
