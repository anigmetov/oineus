"""Differentiable weak-alpha filtration.

Same combinatorics as the alpha complex (built via diode/CGAL), but each
simplex is assigned the squared length of its longest edge (vertices get 0),
recomputed differentiably in the input's framework (torch or jax, via
eagerpy). The squared-distance convention matches cech_delaunay_filtration
so the two are directly comparable, and the longest-edge rule mirrors
Vietoris-Rips restricted to the alpha-complex simplices.
"""

import time
import numpy as np
import eagerpy as epy

from .. import _delaunay_combinatorics, _oineus
from ._backend import concrete_numpy, infer_backend
from ._tensor_utils import real_buffer_for
from .diff_filtration import DiffFiltration


def even_ties_max(stacked):
    """Max over axis 0 of an eagerpy tensor, gradient split evenly among ties.

    eagerpy's max lowers to torch.Tensor.max(dim) on the torch path, whose
    backward sends the whole gradient to the first tied maximum; torch.amax
    and jnp.max (the jax path) split it evenly among ties. Dispatch torch to
    amax so tie gradients match both the pre-eagerpy torch behavior and the
    jax path.
    """
    if infer_backend(stacked.raw) == "torch":
        import torch
        return epy.astensor(torch.amax(stacked.raw, dim=0))
    return epy.max(stacked, axis=0)


def weak_alpha_filtration(points, *, packed: bool = False, print_time: bool = False):
    """Build a differentiable weak-alpha filtration from a point cloud.

    Args:
        points: ``(n, d)`` torch tensor or jax array with ``d in {2, 3}``.
            Differentiable; the returned values are in the same framework.
        packed: Use the compact bit-packed cell encoding for the Delaunay
            combinatorics when the vertex ids fit a 64/128-bit word. The values
            (and gradients) are recomputed here regardless of encoding.
        print_time: If True, print per-stage timings.

    Returns:
        DiffFiltration whose values are squared longest-edge lengths.
    """
    if print_time:
        start = time.time()

    tensor = epy.astensor(points)
    points_np = concrete_numpy(points)
    alpha_fil = _delaunay_combinatorics(points_np, packed=packed)
    if print_time:
        elapsed = time.time() - start
        print(f"alpha_fil construction elapsed: {elapsed:.3f}")

    n0 = alpha_fil.size_in_dimension(0)
    values_in_dim = [epy.zeros(tensor, n0)]

    for dim in range(1, alpha_fil.max_dim + 1):
        if print_time:
            start_dim = time.time()

        if dim == 1:
            edges = alpha_fil.get_edges().astype(np.int64)
            values = ((tensor[edges[:, 0]] - tensor[edges[:, 1]]) ** 2).sum(axis=1)
        elif dim == 2:
            tri = alpha_fil.get_triangles().astype(np.int64)
            p0 = tensor[tri[:, 0]]
            p1 = tensor[tri[:, 1]]
            p2 = tensor[tri[:, 2]]
            d01 = ((p0 - p1) ** 2).sum(axis=1)
            d02 = ((p0 - p2) ** 2).sum(axis=1)
            d12 = ((p1 - p2) ** 2).sum(axis=1)
            values = even_ties_max(epy.stack([d01, d02, d12], axis=0))
        elif dim == 3:
            tet = alpha_fil.get_tetrahedra().astype(np.int64)
            p0 = tensor[tet[:, 0]]
            p1 = tensor[tet[:, 1]]
            p2 = tensor[tet[:, 2]]
            p3 = tensor[tet[:, 3]]
            d01 = ((p0 - p1) ** 2).sum(axis=1)
            d02 = ((p0 - p2) ** 2).sum(axis=1)
            d03 = ((p0 - p3) ** 2).sum(axis=1)
            d12 = ((p1 - p2) ** 2).sum(axis=1)
            d13 = ((p1 - p3) ** 2).sum(axis=1)
            d23 = ((p2 - p3) ** 2).sum(axis=1)
            values = even_ties_max(epy.stack([d01, d02, d03, d12, d13, d23], axis=0))
        else:
            raise RuntimeError(f"weak_alpha_filtration: dim={dim} not supported")

        if print_time:
            elapsed = time.time() - start_dim
            print(f"dim {dim} weak-alpha values elapsed: {elapsed:.3f}")

        values_in_dim.append(values)

    if print_time:
        start = time.time()
    cd_vals = epy.concatenate(values_in_dim)
    # contiguous buffer in the filtration's Real dtype -- read directly by set_values
    alpha_fil.set_values(real_buffer_for(alpha_fil, cd_vals.raw))
    if print_time:
        elapsed = time.time() - start
        print(f"set values elapsed: {elapsed:.3f}")

    if print_time:
        start = time.time()
    sorted_vals = epy.concatenate([epy.sort(vals) for vals in values_in_dim]).raw
    if print_time:
        elapsed = time.time() - start
        print(f"sort values elapsed: {elapsed:.3f}")

    alpha_fil.kind = _oineus.FiltrationKind.WeakAlpha
    return DiffFiltration(alpha_fil, sorted_vals)
