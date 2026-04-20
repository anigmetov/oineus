"""Differentiable Wasserstein distance for persistence diagrams."""

import torch
import numpy as np
from .. import _oineus
from .wasserstein_utils import _project_to_diagonal, _split_finite_essential, _match_essential_1d


def wasserstein_cost(
    dgm_a: torch.Tensor,
    dgm_b: torch.Tensor,
    wasserstein_q: float = 1.0,
    wasserstein_delta: float = 0.05,
    ignore_inf_points: bool = True,
    internal_p: float = float('inf')
) -> torch.Tensor:
    """
    Compute differentiable Wasserstein cost between two persistence diagrams.

    The cost C satisfies: Wasserstein_q distance = C^(1/q)

    Args:
        dgm_a: (N, 2) tensor of (birth, death) points
        dgm_b: (M, 2) tensor of (birth, death) points
        wasserstein_q: Wasserstein power parameter (default 1.0 for W_1)
        wasserstein_delta: Relative error for Hera approximation (default 0.05)
        ignore_inf_points: If True, only consider finite points (default True)
        internal_p: Internal L_p norm for point distances (default inf)

    Returns:
        Scalar tensor with Wasserstein cost^q
        To get distance: wasserstein_cost(...) ** (1/wasserstein_q)

    Notes:
        - Diagonal projections are detached (do not receive gradients)
        - Essential points (with inf coordinates) are matched separately if ignore_inf_points=False
        - Gradients flow through matched finite points

    Example:
        >>> import torch
        >>> import oineus.diff as oin_diff
        >>> dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], requires_grad=True)
        >>> dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]])
        >>> cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0)
        >>> cost.backward()
        >>> print(dgm_a.grad)  # Gradients flow through matched points
    """
    # Split into finite and essential
    fin_a, ess_a = _split_finite_essential(dgm_a)
    fin_b, ess_b = _split_finite_essential(dgm_b)

    total_cost = torch.tensor(0.0, dtype=dgm_a.dtype, device=dgm_a.device)

    # Handle essential points if requested
    if not ignore_inf_points:
        ess_names = ["(finite, +inf)", "(finite, -inf)", "(+inf, finite)", "(-inf, finite)"]
        for coords1, coords2, name in zip(ess_a, ess_b, ess_names):
            if len(coords1) != len(coords2):
                raise ValueError(
                    f"Essential point cardinalities must match. "
                    f"Got {len(coords1)} and {len(coords2)} points with {name}."
                )
            if len(coords1) > 0:
                # Compute 1D matching cost
                cost_1d = _match_essential_1d(coords1, coords2)
                # _match_essential_1d returns sum of |a - b|, need to raise to power q
                if wasserstein_q != 1.0:
                    cost_1d = cost_1d ** wasserstein_q
                total_cost = total_cost + cost_1d

    # Handle finite points
    if len(fin_a) == 0 and len(fin_b) == 0:
        return total_cost

    # Convert finite diagrams to numpy for C++ matching
    fin_a_np = fin_a.detach().cpu().numpy()
    fin_b_np = fin_b.detach().cpu().numpy()

    # Convert internal_p: Hera uses -1.0 to represent L_inf norm
    internal_p_for_hera = -1.0 if np.isinf(internal_p) else internal_p

    # Compute matching using C++
    matching = _oineus.wasserstein_matching_finite(
        fin_a_np, fin_b_np,
        wasserstein_q=wasserstein_q,
        wasserstein_delta=wasserstein_delta,
        internal_p=internal_p_for_hera
    )

    # 1. Finite-to-finite matching
    if len(matching.a_to_b) > 0:
        idx_a = torch.tensor(matching.a_to_b, dtype=torch.long, device=fin_a.device)
        idx_b = torch.tensor(matching.b_from_a, dtype=torch.long, device=fin_b.device)

        pts_a = fin_a[idx_a]
        pts_b = fin_b[idx_b]

        # Compute L_p distance between matched points
        if np.isinf(internal_p):
            # L_inf norm: max absolute difference
            dists = torch.max(torch.abs(pts_a - pts_b), dim=1)[0]
        else:
            # L_p norm
            dists = torch.sum(torch.abs(pts_a - pts_b) ** internal_p, dim=1) ** (1.0 / internal_p)

        total_cost = total_cost + torch.sum(dists ** wasserstein_q)

    # 2. Finite-to-diagonal matching (dgm_a points)
    if len(matching.a_to_diag) > 0:
        idx_a = torch.tensor(matching.a_to_diag, dtype=torch.long, device=fin_a.device)
        pts_a = fin_a[idx_a]

        # Diagonal projection: ((b+d)/2, (b+d)/2) - DETACHED!
        diag_proj = _project_to_diagonal(pts_a).detach()

        # Distance from point to its diagonal projection
        if np.isinf(internal_p):
            dists = torch.max(torch.abs(pts_a - diag_proj), dim=1)[0]
        else:
            dists = torch.sum(torch.abs(pts_a - diag_proj) ** internal_p, dim=1) ** (1.0 / internal_p)

        total_cost = total_cost + torch.sum(dists ** wasserstein_q)

    # 3. Finite-to-diagonal matching (dgm_b points)
    if len(matching.b_to_diag) > 0:
        idx_b = torch.tensor(matching.b_to_diag, dtype=torch.long, device=fin_b.device)
        pts_b = fin_b[idx_b]

        diag_proj = _project_to_diagonal(pts_b).detach()

        if np.isinf(internal_p):
            dists = torch.max(torch.abs(pts_b - diag_proj), dim=1)[0]
        else:
            dists = torch.sum(torch.abs(pts_b - diag_proj) ** internal_p, dim=1) ** (1.0 / internal_p)

        total_cost = total_cost + torch.sum(dists ** wasserstein_q)

    return total_cost
