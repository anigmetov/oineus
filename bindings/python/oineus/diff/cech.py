import numpy as np

try:
    import torch
except ImportError:  # torch-only module; guarded at call time
    torch = None

from .. import _oineus
from .. import cech_filtration as _nondiff_cech_filtration
from ._backend import require_torch
from ._tensor_utils import real_buffer_for
from .cech_delaunay import triangle_meb, tetrahedron_meb
from .diff_filtration import DiffFiltration


def cech_filtration(points, max_dim: int = -1, max_radius: float = -1.0,
                    eps: float = 0.0, *, n_threads: int = 1) -> DiffFiltration:
    """Build a differentiable full Cech filtration from a point cloud.

    The combinatorics (all simplices up to max_dim whose MEB radius is at
    most max_radius) come from the non-differentiable oineus.cech_filtration;
    the filtration values are recomputed differentiably as squared minimum
    enclosing ball radii, so gradients flow back to ``points`` through all
    support points of each ball.

    Unlike cech_delaunay_filtration this enumerates ALL simplices, not just
    Delaunay ones: with the default max_radius the number of q-simplices is
    C(n, q+1), so pass an explicit smaller max_radius beyond a few hundred
    points. Diagrams agree with cech_delaunay_filtration in dimensions
    0 .. max_dim-1 (both compute union-of-balls persistence); the
    dimension-max_dim diagram is skeleton-truncated and unreliable.

    torch-only for now: raises ImportError without torch and TypeError for
    non-torch (e.g. jax) point clouds.

    Args:
        points: ``(n, d)`` torch.Tensor with ``d in {2, 3}``, pairwise
            distinct. Differentiable.
        max_dim: Largest simplex dimension; default d, must be <= d.
        max_radius: Unsquared radius threshold; simplices with squared MEB
            radius <= max_radius**2 are kept. Default: enclosing radius.
        eps: Small value for numerical stability in the MEB computation.
        n_threads: Threads used inside the Filtration constructor.

    Returns:
        DiffFiltration whose values are squared MEB radii.
    """
    require_torch(points, "cech_filtration")

    points_np = points.detach().cpu().numpy()
    # single source of truth for enumeration, pruning and closure; its numpy
    # values are provisional and get overwritten by the torch recomputation
    fil = _nondiff_cech_filtration(points_np, max_dim=max_dim, max_radius=max_radius,
                                   eps=eps, n_threads=n_threads)

    values_in_dim = [torch.zeros(fil.size_in_dimension(0), requires_grad=True, device=points.device)]
    for dim in range(1, fil.max_dim + 1):
        simplices = torch.LongTensor(fil.get_simplices_as_arr(dim).astype(np.int64))
        ps = [points[simplices[:, k]] for k in range(dim + 1)]
        if dim == 1:
            radii_sq = 0.25 * torch.sum((ps[0] - ps[1]) ** 2, dim=1)
        elif dim == 2:
            _, radii_sq = triangle_meb(ps[0], ps[1], ps[2], eps)
        elif dim == 3:
            _, radii_sq = tetrahedron_meb(ps[0], ps[1], ps[2], ps[3], eps)
        else:
            raise RuntimeError("Dimension not supported")
        assert simplices.shape[0] == radii_sq.shape[0]
        values_in_dim.append(radii_sq)

    # re-sort the filtration on the torch-derived values so the internal
    # order stays consistent with the per-dim sorted tensor below even
    # where numpy and torch arithmetic differ in the last ulp
    fil.set_values(real_buffer_for(fil, torch.cat(values_in_dim)))

    sorted_vals = torch.cat([torch.sort(vals)[0] for vals in values_in_dim])
    fil.kind = _oineus.FiltrationKind.Cech
    return DiffFiltration(fil, sorted_vals)
