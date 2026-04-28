"""Differentiable weak-alpha filtration.

Same combinatorics as the alpha complex (built via diode/CGAL), but each
simplex is assigned the squared length of its longest edge (vertices get 0).
The squared-distance convention matches cech_delaunay_filtration so the two
are directly comparable, and the longest-edge rule mirrors Vietoris-Rips
restricted to the alpha-complex simplices.
"""

import time
import numpy as np
import torch

from .. import _alpha_shapes_filtration, _oineus
from .diff_filtration import DiffFiltration


def weak_alpha_filtration(points, print_time: bool = False):
    """Build a differentiable weak-alpha filtration from a point cloud.

    Args:
        points: ``(n, d)`` torch.Tensor with ``d in {2, 3}``. Differentiable.
        print_time: If True, print per-stage timings.

    Returns:
        DiffFiltration whose values are squared longest-edge lengths.
    """
    if print_time:
        start = time.time()

    points_np = points.detach().cpu().numpy()
    alpha_fil = _alpha_shapes_filtration(points_np)
    if print_time:
        elapsed = time.time() - start
        print(f"alpha_fil construction elapsed: {elapsed:.3f}")

    n0 = alpha_fil.size_in_dimension(0)
    values_in_dim = [
        torch.zeros(n0, requires_grad=True, device=points.device, dtype=points.dtype)
    ]

    for dim in range(1, alpha_fil.max_dim() + 1):
        if print_time:
            start_dim = time.time()

        if dim == 1:
            edges = torch.LongTensor(alpha_fil.get_edges().astype(np.uint64))
            values = torch.sum((points[edges[:, 0]] - points[edges[:, 1]]) ** 2, dim=1)
        elif dim == 2:
            tri = torch.LongTensor(alpha_fil.get_triangles().astype(np.uint64))
            p0 = points[tri[:, 0]]
            p1 = points[tri[:, 1]]
            p2 = points[tri[:, 2]]
            d01 = torch.sum((p0 - p1) ** 2, dim=1)
            d02 = torch.sum((p0 - p2) ** 2, dim=1)
            d12 = torch.sum((p1 - p2) ** 2, dim=1)
            values = torch.amax(torch.stack([d01, d02, d12], dim=0), dim=0)
        elif dim == 3:
            tet = torch.LongTensor(alpha_fil.get_tetrahedra().astype(np.uint64))
            p0 = points[tet[:, 0]]
            p1 = points[tet[:, 1]]
            p2 = points[tet[:, 2]]
            p3 = points[tet[:, 3]]
            d01 = torch.sum((p0 - p1) ** 2, dim=1)
            d02 = torch.sum((p0 - p2) ** 2, dim=1)
            d03 = torch.sum((p0 - p3) ** 2, dim=1)
            d12 = torch.sum((p1 - p2) ** 2, dim=1)
            d13 = torch.sum((p1 - p3) ** 2, dim=1)
            d23 = torch.sum((p2 - p3) ** 2, dim=1)
            values = torch.amax(torch.stack([d01, d02, d03, d12, d13, d23], dim=0), dim=0)
        else:
            raise RuntimeError(f"weak_alpha_filtration: dim={dim} not supported")

        if print_time:
            elapsed = time.time() - start_dim
            print(f"dim {dim} weak-alpha values elapsed: {elapsed:.3f}")

        values_in_dim.append(values)

    if print_time:
        start = time.time()
    cd_vals = torch.cat(values_in_dim)
    cd_vals_list = [float(x) for x in cd_vals.clone().detach().cpu()]
    alpha_fil.set_values(cd_vals_list)
    if print_time:
        elapsed = time.time() - start
        print(f"set values elapsed: {elapsed:.3f}")

    if print_time:
        start = time.time()
    sorted_vals = torch.cat([torch.sort(vals)[0] for vals in values_in_dim])
    if print_time:
        elapsed = time.time() - start
        print(f"sort values elapsed: {elapsed:.3f}")

    alpha_fil.kind = _oineus.FiltrationKind.WeakAlpha
    return DiffFiltration(alpha_fil, sorted_vals)
