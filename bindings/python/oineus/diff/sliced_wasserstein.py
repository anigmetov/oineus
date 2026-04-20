import torch
import numpy as np

from .wasserstein_utils import _project_to_diagonal, _split_finite_essential, _match_essential_1d


def _sample_unit_directions(n_directions, device, dtype):
    """Sample n_directions unit vectors on the unit circle (actually half-circle is enough)."""
    # Sample angles in [0, pi) for half-circle
    angles = torch.rand(n_directions, device=device, dtype=dtype) * np.pi
    # Convert to unit vectors
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def _compute_slice_cost_standard(fin1, fin2, u):
    """
    Compute sliced Wasserstein cost for a single direction (standard version).

    Args:
        fin1: (n1, 2) finite points from dgm1
        fin2: (n2, 2) finite points from dgm2
        u: (2,) unit direction vector

    Returns:
        Scalar cost for this slice
    """
    n1, n2 = len(fin1), len(fin2)

    if n1 == 0 and n2 == 0:
        return torch.tensor(0.0, dtype=u.dtype, device=u.device)

    # Project finite points onto direction u
    proj1 = (fin1 @ u) if n1 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)
    proj2 = (fin2 @ u) if n2 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)

    # Add diagonal projections of opposite diagram
    if n2 > 0:
        diag_proj2 = _project_to_diagonal(fin2)
        proj1_diag = diag_proj2 @ u
    else:
        proj1_diag = torch.tensor([], dtype=u.dtype, device=u.device)

    if n1 > 0:
        diag_proj1 = _project_to_diagonal(fin1)
        proj2_diag = diag_proj1 @ u
    else:
        proj2_diag = torch.tensor([], dtype=u.dtype, device=u.device)

    # Augmented 1D measures
    L1 = torch.cat([proj1, proj1_diag]) if n1 > 0 or n2 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)
    L2 = torch.cat([proj2, proj2_diag]) if n2 > 0 or n1 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)

    if len(L1) == 0:
        return torch.tensor(0.0, dtype=u.dtype, device=u.device)

    # Sort
    L1_sorted = torch.sort(L1)[0]
    L2_sorted = torch.sort(L2)[0]

    # Match by rank and compute cost
    return torch.sum(torch.abs(L1_sorted - L2_sorted))


def _compute_slice_cost_corrected(fin1, fin2, u):
    """
    Compute diagonal-corrected sliced Wasserstein cost for a single direction.

    Args:
        fin1: (n1, 2) finite points from dgm1
        fin2: (n2, 2) finite points from dgm2
        u: (2,) unit direction vector

    Returns:
        Scalar cost for this slice
    """
    n1, n2 = len(fin1), len(fin2)

    # Handle empty diagram cases specially to avoid indexing issues
    if n1 == 0 and n2 == 0:
        return torch.tensor(0.0, dtype=u.dtype, device=u.device)

    if n1 == 0:
        # Cost: distance from each point in fin2 to its own diagonal projection
        diag_proj2 = _project_to_diagonal(fin2).detach()
        proj2 = fin2 @ u
        proj2_diag = diag_proj2 @ u
        return torch.sum(torch.abs(proj2 - proj2_diag))

    if n2 == 0:
        # Cost: distance from each point in fin1 to its own diagonal projection
        diag_proj1 = _project_to_diagonal(fin1).detach()
        proj1 = fin1 @ u
        proj1_diag = diag_proj1 @ u
        return torch.sum(torch.abs(proj1 - proj1_diag))

    # Project finite points onto direction u
    proj1 = (fin1 @ u) if n1 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)
    proj2 = (fin2 @ u) if n2 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)

    # Diagonal projections - DETACHED so they don't contribute to gradients
    if n1 > 0:
        diag_proj1 = _project_to_diagonal(fin1).detach()
        proj1_self_diag = diag_proj1 @ u
        proj2_diag_from_1 = diag_proj1 @ u
    else:
        proj1_self_diag = torch.tensor([], dtype=u.dtype, device=u.device)
        proj2_diag_from_1 = torch.tensor([], dtype=u.dtype, device=u.device)

    if n2 > 0:
        diag_proj2 = _project_to_diagonal(fin2).detach()
        proj2_self_diag = diag_proj2 @ u
        proj1_diag_from_2 = diag_proj2 @ u
    else:
        proj2_self_diag = torch.tensor([], dtype=u.dtype, device=u.device)
        proj1_diag_from_2 = torch.tensor([], dtype=u.dtype, device=u.device)

    # Build augmented lists
    L1 = torch.cat([proj1, proj1_diag_from_2]) if n1 > 0 or n2 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)
    L2 = torch.cat([proj2, proj2_diag_from_1]) if n2 > 0 or n1 > 0 else torch.tensor([], dtype=u.dtype, device=u.device)

    if len(L1) == 0:
        return torch.tensor(0.0, dtype=u.dtype, device=u.device)

    # Sort and track indices
    L1_sorted, L1_indices = torch.sort(L1)
    L2_sorted, L2_indices = torch.sort(L2)

    # Vectorized cost computation using torch.where (vmap-compatible)
    # Determine which points are diagonal projections
    is_diag1 = (L1_indices >= n1)  # L1 points that are diagonal projections from dgm2
    is_diag2 = (L2_indices >= n2)  # L2 points that are diagonal projections from dgm1

    # Case 1: both diagonal -> cost = 0
    # Case 2: L1 is diagonal, L2 is real -> cost = |L2_real - L2_self_diag|
    # Case 3: L1 is real, L2 is diagonal -> cost = |L1_real - L1_self_diag|
    # Case 4: both real -> cost = |L1 - L2|

    # Default: standard matching cost (Case 4)
    costs = torch.abs(L1_sorted - L2_sorted)

    # Case 3: L1 is real (from dgm1), L2 is diagonal (from dgm1)
    # Cost should be |L1_real - L1_self_diag|
    # Use torch.where to avoid boolean indexing (vmap-compatible)
    real_idx1 = torch.clamp(L1_indices, 0, n1 - 1)  # Clamp for safe indexing
    cost_case3 = torch.abs(proj1[real_idx1] - proj1_self_diag[real_idx1])
    costs = torch.where(~is_diag1 & is_diag2, cost_case3, costs)

    # Case 2: L1 is diagonal (from dgm2), L2 is real (from dgm2)
    # Cost should be |L2_real - L2_self_diag|
    real_idx2 = torch.clamp(L2_indices, 0, n2 - 1)  # Clamp for safe indexing
    cost_case2 = torch.abs(proj2[real_idx2] - proj2_self_diag[real_idx2])
    costs = torch.where(is_diag1 & ~is_diag2, cost_case2, costs)

    # Case 1: both diagonal -> cost = 0
    costs = torch.where(is_diag1 & is_diag2, torch.zeros_like(costs), costs)

    return torch.sum(costs)


def sliced_wasserstein_distance(dgm1, dgm2, n_directions=100, ignore_inf_points=False):
    """
    Sliced Wasserstein distance between two persistence diagrams.

    This is the standard sliced Wasserstein where diagonal projections participate
    in gradients. When a point p from dgm1 is matched to diag_proj(q) from dgm2,
    both p and q receive gradients.

    Args:
        dgm1: (N, 2) tensor of persistence diagram points (birth, death)
        dgm2: (M, 2) tensor of persistence diagram points (birth, death)
        n_directions: Number of random projection directions
        ignore_inf_points: If True, only consider finite points

    Returns:
        Scalar tensor with the sliced Wasserstein distance
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return torch.tensor(0.0, dtype=dgm1.dtype, device=dgm1.device)

    # Split into finite and essential
    fin1, ess1 = _split_finite_essential(dgm1)
    fin2, ess2 = _split_finite_essential(dgm2)

    total_cost = torch.tensor(0.0, dtype=dgm1.dtype, device=dgm1.device)

    # Handle essential points if requested
    if not ignore_inf_points:
        ess_names = ["(finite, +inf)", "(finite, -inf)", "(+inf, finite)", "(-inf, finite)"]
        for coords1, coords2, name in zip(ess1, ess2, ess_names):
            if len(coords1) != len(coords2):
                raise ValueError(
                    f"Essential point cardinalities must match. "
                    f"Got {len(coords1)} and {len(coords2)} points with {name}."
                )
            if len(coords1) > 0:
                total_cost = total_cost + _match_essential_1d(coords1, coords2)

    # Handle finite points
    if len(fin1) == 0 and len(fin2) == 0:
        return total_cost

    # Sample random directions
    directions = _sample_unit_directions(n_directions, dgm1.device, dgm1.dtype)

    # Vectorized computation over directions using vmap
    slice_costs = torch.vmap(lambda u: _compute_slice_cost_standard(fin1, fin2, u))(directions)
    total_cost = total_cost + slice_costs.mean()

    return total_cost


def sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, n_directions=100, ignore_inf_points=False):
    """
    Diagonal-corrected sliced Wasserstein distance.

    This version differs from standard sliced Wasserstein in two ways:
    1. Diagonal projections do NOT contribute to gradients (they are detached)
    2. When a point p is matched to a diagonal projection, the cost is computed
       as if p is matched to its own diagonal projection (not the other diagram's)
    3. Matchings between two diagonal projections contribute zero cost

    This matches the behavior of standard Wasserstein distance where each point
    is matched to its own diagonal projection with cost = persistence/2.

    Args:
        dgm1: (N, 2) tensor of persistence diagram points (birth, death)
        dgm2: (M, 2) tensor of persistence diagram points (birth, death)
        n_directions: Number of random projection directions
        ignore_inf_points: If True, only consider finite points

    Returns:
        Scalar tensor with the diagonal-corrected sliced Wasserstein distance
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return torch.tensor(0.0, dtype=dgm1.dtype, device=dgm1.device)

    # Split into finite and essential
    fin1, ess1 = _split_finite_essential(dgm1)
    fin2, ess2 = _split_finite_essential(dgm2)

    total_cost = torch.tensor(0.0, dtype=dgm1.dtype, device=dgm1.device)

    # Handle essential points if requested
    if not ignore_inf_points:
        ess_names = ["(finite, +inf)", "(finite, -inf)", "(+inf, finite)", "(-inf, finite)"]
        for coords1, coords2, name in zip(ess1, ess2, ess_names):
            if len(coords1) != len(coords2):
                raise ValueError(
                    f"Essential point cardinalities must match. "
                    f"Got {len(coords1)} and {len(coords2)} points with {name}."
                )
            if len(coords1) > 0:
                total_cost = total_cost + _match_essential_1d(coords1, coords2)

    # Handle finite points
    if len(fin1) == 0 and len(fin2) == 0:
        return total_cost

    # Sample random directions
    directions = _sample_unit_directions(n_directions, dgm1.device, dgm1.dtype)

    # Vectorized computation over directions using vmap
    slice_costs = torch.vmap(lambda u: _compute_slice_cost_corrected(fin1, fin2, u))(directions)
    total_cost = total_cost + slice_costs.mean()

    return total_cost
