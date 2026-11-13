import eagerpy as epy

from .. import vr_filtration as non_diff_vr_filtration
from .diff_filtration import DiffFiltration
from ._tensor_utils import tensor_to_real_numpy


def vr_filtration(data, from_pwdists: bool = False, max_dim: int = -1,
                  max_diameter: float = -1.0, eps=1e-6, *, packed: bool = False,
                  n_threads=1) -> DiffFiltration:
    """Build a differentiable Vietoris-Rips filtration from a point cloud or distance matrix.

    Enumerates simplices and their critical defining edges via C++
    :func:`oineus.vr_filtration`, then recomputes critical edge lengths in the
    input tensor's framework (PyTorch or JAX). For point-cloud inputs, lengths
    are smoothed as ``sqrt(||x_i - x_j||^2 + eps)`` to avoid zero-distance
    gradient singularities (including on vertices, where ``i == j``).

    Args:
        data: ``(n, d)`` point cloud tensor/array, or ``(n, n)`` pairwise
            distance tensor/array when ``from_pwdists=True`` (PyTorch or JAX).
        from_pwdists: If True, treat ``data`` as a pairwise distance matrix.
        max_dim: Maximal simplex dimension to enumerate (default: ambient
            dimension ``d`` for point clouds; required if ``from_pwdists=True``).
        max_diameter: Diameter cutoff; negative values default to enclosing radius.
        eps: Smoothing constant added inside ``sqrt(dist^2 + eps)`` when
            ``from_pwdists=False``.
        packed: If True, use compact bit-packed simplex encoding when vertex ids
            fit a 64- or 128-bit word.
        n_threads: Threads for C++ filtration sorting.

    Returns:
        :class:`DiffFiltration` with differentiable simplex diameters.
    """
    data = epy.astensor(data)
    data_np = tensor_to_real_numpy(data)
    assert data.ndim == 2

    # packed=True builds the bit-packed PackedSimplexFiltration when the tier fits (else fat);
    # the critical-edge array is encoding-independent (vertex-id pairs aligned to the sorted
    # cells), so the differentiable distance gather below is unchanged. oineus.diff.TopologyOptimizer
    # dispatches the packed type and mapping_cylinder_filtration fattens packed under-filtrations as
    # needed. It is opt-in for the differentiable factory (the non-diff oineus.vr_filtration defaults
    # to packed) because some low-level diff entry points (compute_partial_u_rows, the raw per-cell
    # optimizer ctor) are still fat-only.
    fil, edges = non_diff_vr_filtration(
        data=data_np,
        from_pwdists=from_pwdists,
        with_critical_edges=True,
        max_dim=max_dim,
        max_diameter=max_diameter,
        packed=packed,
        n_threads=n_threads,
    )

    if not from_pwdists:
        sqdists = epy.sum((data[edges[:, 0].flatten()] - data[edges[:, 1].flatten()]) ** 2, axis=1) + eps
        diff_dists = epy.sqrt(sqdists).raw
        return DiffFiltration(fil, diff_dists)
    else:
        edges = epy.astensor(edges)
        diff_dists = data[edges[:, 0], edges[:, 1]].raw
        return DiffFiltration(fil, diff_dists)
