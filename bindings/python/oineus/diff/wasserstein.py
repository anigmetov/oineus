"""Differentiable Wasserstein cost for persistence diagrams.

Single function that calls the new detailed Hera matching once on the full
diagrams (essentials included), then walks the bucketed result and rebuilds
the cost in torch so gradients flow through every matched pair (finite-to-
finite, finite-to-diagonal, and essential-to-essential).
"""

import numpy as np
import torch

from .. import _oineus
from .._dtype import as_real_numpy

# Mapping from essential family attribute name to the index of the finite
# coordinate axis (0 = birth, 1 = death). Used to compute the per-pair
# 1D distance: the infinite axis matches itself with cost 0.
_ESSENTIAL_FINITE_AXIS = (
    ("inf_death",     0),
    ("neg_inf_death", 0),
    ("inf_birth",     1),
    ("neg_inf_birth", 1),
)


def wasserstein_cost(
    dgm_a: torch.Tensor,
    dgm_b: torch.Tensor,
    wasserstein_q: float = 1.0,
    wasserstein_delta: float = 0.05,
    ignore_inf_points: bool = True,
    internal_p: float = float("inf"),
) -> torch.Tensor:
    """Differentiable Wasserstein cost between two persistence diagrams.

    Returns ``cost = sum_pair dist(p_a, p_b) ** wasserstein_q`` so that
    ``Wasserstein_q distance == cost ** (1 / wasserstein_q)``.

    Args:
        dgm_a: ``(N, 2)`` tensor of (birth, death) points.
        dgm_b: ``(M, 2)`` tensor of (birth, death) points.
        wasserstein_q: Wasserstein power (default 1.0 → W_1).
        wasserstein_delta: Hera relative-error parameter (must be > 0).
        ignore_inf_points: If True, drop essential (±inf) points before
            matching. If False, every essential family must have equal
            cardinalities on both sides; the matching pairs them by
            sorted-rank of the finite coordinate.
        internal_p: Ground metric in the (birth, death) plane. ``inf``
            selects L_∞.

    Returns:
        Scalar tensor with the cost. Gradients flow through every matched
        finite point on both sides (and through the finite coord of every
        matched essential). Diagonal projections are detached.
    """
    device = dgm_a.device
    dtype  = dgm_a.dtype

    # Convert to numpy and call the bucketed Hera matching once.
    dgm_a_np = as_real_numpy(dgm_a.detach().cpu().numpy())
    dgm_b_np = as_real_numpy(dgm_b.detach().cpu().numpy())
    internal_p_hera = -1.0 if np.isinf(internal_p) else internal_p

    matching = _oineus.wasserstein_matching_detailed(
        dgm_a_np, dgm_b_np,
        wasserstein_q=wasserstein_q,
        wasserstein_delta=wasserstein_delta,
        internal_p=internal_p_hera,
        ignore_inf_points=ignore_inf_points,
    )

    total = torch.zeros((), dtype=dtype, device=device)

    def _pair_dist(pts_a, pts_b):
        diff = pts_a - pts_b
        if np.isinf(internal_p):
            return torch.max(torch.abs(diff), dim=1)[0]
        return torch.sum(torch.abs(diff) ** internal_p, dim=1) ** (1.0 / internal_p)

    # 1. finite-to-finite
    ftf = matching.finite_to_finite
    if ftf.shape[0] > 0:
        ia = torch.from_numpy(ftf[:, 0]).to(device).long()
        ib = torch.from_numpy(ftf[:, 1]).to(device).long()
        total = total + torch.sum(_pair_dist(dgm_a[ia], dgm_b[ib]) ** wasserstein_q)

    # 2. finite-to-diagonal (a side)
    if matching.a_to_diagonal.shape[0] > 0:
        ia = torch.from_numpy(matching.a_to_diagonal).to(device).long()
        pts = dgm_a[ia]
        mid = ((pts[:, 0] + pts[:, 1]) / 2.0).detach()
        diag_proj = torch.stack([mid, mid], dim=1)
        total = total + torch.sum(_pair_dist(pts, diag_proj) ** wasserstein_q)

    # 3. finite-to-diagonal (b side)
    if matching.b_to_diagonal.shape[0] > 0:
        ib = torch.from_numpy(matching.b_to_diagonal).to(device).long()
        pts = dgm_b[ib]
        mid = ((pts[:, 0] + pts[:, 1]) / 2.0).detach()
        diag_proj = torch.stack([mid, mid], dim=1)
        total = total + torch.sum(_pair_dist(pts, diag_proj) ** wasserstein_q)

    # 4. essentials, per family — the shared infinite coord contributes 0
    # to the ground metric for any internal_p, so cost is just
    # |finite_a - finite_b| ** wasserstein_q on the finite axis.
    for name, axis in _ESSENTIAL_FINITE_AXIS:
        pairs = getattr(matching.essential, name)
        if pairs.shape[0] == 0:
            continue
        ia = torch.from_numpy(pairs[:, 0]).to(device).long()
        ib = torch.from_numpy(pairs[:, 1]).to(device).long()
        d = torch.abs(dgm_a[ia, axis] - dgm_b[ib, axis])
        total = total + torch.sum(d ** wasserstein_q)

    return total
