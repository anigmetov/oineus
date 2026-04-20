"""Shared utilities for Wasserstein and sliced Wasserstein distances."""

import torch


def _project_to_diagonal(points):
    """Project points (b, d) to diagonal as ((b+d)/2, (b+d)/2)."""
    if len(points) == 0:
        return points
    t = (points[:, 0] + points[:, 1]) / 2.0
    return torch.stack([t, t], dim=1)


def _split_finite_essential(dgm):
    """
    Split diagram into finite points and finite coordinates of essential points.

    Valid points have at most one infinite coordinate.
    Points with both coordinates infinite are discarded.

    Returns:
        finite: (N_fin, 2) tensor of finite points
        essential_coords: tuple of 4 tensors (1D), each containing finite coordinates:
            [0] (finite, +inf) -> birth values
            [1] (finite, -inf) -> birth values
            [2] (+inf, finite) -> death values
            [3] (-inf, finite) -> death values
    """
    empty_1d = torch.tensor([], dtype=dgm.dtype, device=dgm.device)

    if len(dgm) == 0:
        return torch.zeros((0, 2), dtype=dgm.dtype, device=dgm.device), (empty_1d, empty_1d, empty_1d, empty_1d)

    births = dgm[:, 0]
    deaths = dgm[:, 1]

    is_finite = torch.isfinite(births) & torch.isfinite(deaths)
    finite = dgm[is_finite]

    # Filter out invalid points where both coordinates are infinite
    both_inf = ~torch.isfinite(births) & ~torch.isfinite(deaths)
    valid_mask = ~both_inf

    # Essential points: exactly one coordinate is infinite
    is_ess = ~is_finite & valid_mask
    ess = dgm[is_ess]

    if len(ess) == 0:
        return finite, (empty_1d, empty_1d, empty_1d, empty_1d)

    ess_births = ess[:, 0]
    ess_deaths = ess[:, 1]

    # Extract finite coordinates for each essential category
    # (finite, +inf) -> births
    bfin_pinf = torch.isfinite(ess_births) & torch.isposinf(ess_deaths)
    coords_bfin_pinf = ess_births[bfin_pinf] if bfin_pinf.any() else empty_1d

    # (finite, -inf) -> births
    bfin_ninf = torch.isfinite(ess_births) & torch.isneginf(ess_deaths)
    coords_bfin_ninf = ess_births[bfin_ninf] if bfin_ninf.any() else empty_1d

    # (+inf, finite) -> deaths
    pinf_dfin = torch.isposinf(ess_births) & torch.isfinite(ess_deaths)
    coords_pinf_dfin = ess_deaths[pinf_dfin] if pinf_dfin.any() else empty_1d

    # (-inf, finite) -> deaths
    ninf_dfin = torch.isneginf(ess_births) & torch.isfinite(ess_deaths)
    coords_ninf_dfin = ess_deaths[ninf_dfin] if ninf_dfin.any() else empty_1d

    return finite, (coords_bfin_pinf, coords_bfin_ninf, coords_pinf_dfin, coords_ninf_dfin)


def _match_essential_1d(ess1: torch.Tensor, ess2: torch.Tensor, q: float = 1.0) -> torch.Tensor:
    """
    Match essential points in 1D.

    Args:
        ess1: 1D tensor of finite coordinate values from dgm1 essential points
        ess2: 1D tensor of finite coordinate values from dgm2 essential points
        q: Wasserstein power parameter (default 1.0)

    Returns:
        Scalar tensor with the matching cost: sum_i |a_i - b_i|^q

    Both tensors must have the same length (same cardinality of essential points).
    Sorts and matches by rank.
    """
    n1, n2 = len(ess1), len(ess2)
    if n1 != n2:
        raise ValueError(f"Essential point cardinalities must match: {n1} vs {n2}")

    if n1 == 0:
        return torch.tensor(0.0, dtype=ess1.dtype, device=ess1.device)

    sorted1 = torch.sort(ess1)[0]
    sorted2 = torch.sort(ess2)[0]
    return torch.sum(torch.abs(sorted1 - sorted2) ** q)
