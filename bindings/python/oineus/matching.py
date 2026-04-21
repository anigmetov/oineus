"""Wasserstein matching between persistence diagrams."""

import typing
import numpy as np
from . import _oineus


class DiagramMatching:
    """
    Optimal matching between two persistence diagrams for Wasserstein distance.

    All indices refer to positions in the original input diagrams (before any
    internal splitting of finite/essential points).

    Attributes
    ----------
    finite_to_finite : list[tuple[int, int]]
        Pairs of indices (idx_a, idx_b) for finite points matched to each other.
        Each idx_a is an index in dgm_a, idx_b is an index in dgm_b.

    a_to_diagonal : list[int]
        Indices in dgm_a of finite points matched to the diagonal.
        These points are matched to their own diagonal projections.
        Use `point_to_diagonal(dgm_a)` to get all projection coordinates.

    b_to_diagonal : list[int]
        Indices in dgm_b of finite points matched to the diagonal.

    essential_matches : dict[str, list[tuple[int, int]]]
        Matches between essential points, grouped by category.
        Keys are: "(finite, +inf)", "(finite, -inf)", "(+inf, finite)", "(-inf, finite)"
        Values are lists of (idx_a, idx_b) tuples.
        Essential points are matched by sorting their finite coordinates.
        Empty if ignore_inf_points=True.

    cost : float
        Total Wasserstein cost (distance^q).

    distance : float
        Wasserstein distance (cost^(1/q)).

    Examples
    --------
    Plot matching between two diagrams including diagonal projections:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import oineus
    >>>
    >>> # Create diagrams with different number of points
    >>> dgm_a = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
    >>> dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])
    >>>
    >>> matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
    >>>
    >>> # Plot diagrams
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(dgm_a[:, 0], dgm_a[:, 1], c='red', s=100, label='Diagram A')
    >>> ax.scatter(dgm_b[:, 0], dgm_b[:, 1], c='blue', s=100, label='Diagram B')
    >>>
    >>> # Plot finite-to-finite matches
    >>> for idx_a, idx_b in matching.finite_to_finite:
    ...     pt_a = dgm_a[idx_a]
    ...     pt_b = dgm_b[idx_b]
    ...     ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]],
    ...             'k-', alpha=0.5, linewidth=2)
    >>>
    >>> # Get all diagonal projections at once (efficient!)
    >>> projs_a = oineus.point_to_diagonal(dgm_a)
    >>> projs_b = oineus.point_to_diagonal(dgm_b)
    >>>
    >>> # Plot finite-to-diagonal matches using precomputed projections
    >>> for idx in matching.a_to_diagonal:
    ...     ax.scatter([projs_a[idx, 0]], [projs_a[idx, 1]], c='red', marker='x', s=50)
    ...     ax.plot([dgm_a[idx, 0], projs_a[idx, 0]],
    ...             [dgm_a[idx, 1], projs_a[idx, 1]],
    ...             'r--', alpha=0.5, linewidth=2)
    >>>
    >>> for idx in matching.b_to_diagonal:
    ...     ax.scatter([projs_b[idx, 0]], [projs_b[idx, 1]], c='blue', marker='x', s=50)
    ...     ax.plot([dgm_b[idx, 0], projs_b[idx, 0]],
    ...             [dgm_b[idx, 1], projs_b[idx, 1]],
    ...             'b--', alpha=0.5, linewidth=2)
    >>>
    >>> # Plot diagonal
    >>> lim = max(dgm_a.max(), dgm_b.max())
    >>> ax.plot([0, lim], [0, lim], 'k--', alpha=0.2)
    >>> ax.set_xlabel('Birth')
    >>> ax.set_ylabel('Death')
    >>> ax.legend()
    >>> plt.show()
    >>>
    >>> print(f"Wasserstein distance: {matching.distance:.4f}")

    See Also
    --------
    wasserstein_matching : Compute matching between diagrams
    point_to_diagonal : Get diagonal projection coordinates for points
    wasserstein_distance : Compute only the distance (no matching info)
    """

    def __init__(self):
        self.finite_to_finite = []
        self.a_to_diagonal = []
        self.b_to_diagonal = []
        self.essential_matches = {}
        self.cost = 0.0
        self.distance = 0.0

    def __repr__(self):
        return (f"DiagramMatching("
                f"finite_to_finite={len(self.finite_to_finite)}, "
                f"a_to_diagonal={len(self.a_to_diagonal)}, "
                f"b_to_diagonal={len(self.b_to_diagonal)}, "
                f"essential={sum(len(v) for v in self.essential_matches.values())}, "
                f"distance={self.distance:.4f})")


def point_to_diagonal(dgm, indices=None):
    """
    Get diagonal projection coordinates for points in a persistence diagram.

    The diagonal projection of a point (b, d) is ((b+d)/2, (b+d)/2).
    This function efficiently computes projections for multiple points at once.

    Parameters
    ----------
    dgm : array-like or list of DiagramPoint
        Persistence diagram as (n_points, 2) array or list of DiagramPoint objects.

    indices : array-like or None, default=None
        Indices of points to project. If None, projects all points.

    Returns
    -------
    ndarray or list
        If dgm is numpy array: (n, 2) array where n is number of projected points.
        If dgm is list: list of (x, x) tuples.

    Examples
    --------
    Project all points (efficient for plotting):

    >>> import numpy as np
    >>> import oineus
    >>>
    >>> dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
    >>> projs = oineus.point_to_diagonal(dgm)
    >>> projs
    array([[0.5 , 0.5 ],
           [1.25, 1.25],
           [2.25, 2.25]])

    Project specific points:

    >>> projs = oineus.point_to_diagonal(dgm, indices=[0, 2])
    >>> projs
    array([[0.5 , 0.5 ],
           [2.25, 2.25]])

    Use with matching to plot diagonal projections efficiently:

    >>> matching = oineus.wasserstein_matching(dgm_a, dgm_b)
    >>> # Compute all projections once
    >>> projs_a = oineus.point_to_diagonal(dgm_a)
    >>> projs_b = oineus.point_to_diagonal(dgm_b)
    >>>
    >>> # Plot using precomputed projections
    >>> for idx in matching.a_to_diagonal:
    ...     plt.plot([dgm_a[idx, 0], projs_a[idx, 0]],
    ...              [dgm_a[idx, 1], projs_a[idx, 1]], 'r--')

    See Also
    --------
    wasserstein_matching : Compute matching between diagrams
    DiagramMatching : Matching result object
    """
    # Handle different input types
    if isinstance(dgm, np.ndarray):
        # NumPy array - most common case
        if indices is None:
            # Project all points
            b = dgm[:, 0]
            d = dgm[:, 1]
        else:
            # Project specific points
            indices = np.asarray(indices)
            b = dgm[indices, 0]
            d = dgm[indices, 1]

        # Compute projections
        diag_coords = (b + d) / 2.0
        return np.column_stack([diag_coords, diag_coords])

    elif isinstance(dgm, list):
        # List of points or DiagramPoint objects
        if indices is None:
            indices = range(len(dgm))

        result = []
        for i in indices:
            pt = dgm[i]
            if hasattr(pt, 'birth') and hasattr(pt, 'death'):
                # DiagramPoint object
                b, d = pt.birth, pt.death
            else:
                # Assume tuple/list
                b, d = float(pt[0]), float(pt[1])

            diag_coord = (b + d) / 2.0
            result.append((diag_coord, diag_coord))

        return result

    else:
        # Try generic indexing (e.g., Diagrams object)
        # Return as list since these won't be large
        if indices is None:
            indices = range(len(dgm))

        result = []
        for i in indices:
            pt = dgm[i]
            if hasattr(pt, 'birth') and hasattr(pt, 'death'):
                b, d = pt.birth, pt.death
            else:
                b, d = float(pt[0]), float(pt[1])

            diag_coord = (b + d) / 2.0
            result.append((diag_coord, diag_coord))

        return result


def _split_with_indices(dgm):
    """
    Split diagram into finite and essential parts, tracking original indices.

    Private helper for wasserstein_matching().

    Returns
    -------
    finite : ndarray
        (n_finite, 2) array of finite points
    essential : dict
        Maps category -> 1D array of finite coordinates
    finite_indices : list
        Original indices for finite points
    essential_indices : dict
        Maps category -> list of original indices
    """
    finite_pts = []
    finite_idx = []

    essential_coords = {
        "(finite, +inf)": [],
        "(finite, -inf)": [],
        "(+inf, finite)": [],
        "(-inf, finite)": []
    }
    essential_idx = {cat: [] for cat in essential_coords}

    for i, (b, d) in enumerate(dgm):
        if np.isfinite(b) and np.isfinite(d):
            if b != d:  # Skip diagonal points
                finite_pts.append([b, d])
                finite_idx.append(i)
        elif np.isfinite(b) and np.isposinf(d):
            essential_coords["(finite, +inf)"].append(b)
            essential_idx["(finite, +inf)"].append(i)
        elif np.isfinite(b) and np.isneginf(d):
            essential_coords["(finite, -inf)"].append(b)
            essential_idx["(finite, -inf)"].append(i)
        elif np.isposinf(b) and np.isfinite(d):
            essential_coords["(+inf, finite)"].append(d)
            essential_idx["(+inf, finite)"].append(i)
        elif np.isneginf(b) and np.isfinite(d):
            essential_coords["(-inf, finite)"].append(d)
            essential_idx["(-inf, finite)"].append(i)
        # Skip points with both coords infinite or on diagonal

    finite = np.array(finite_pts, dtype=np.float64) if finite_pts else np.zeros((0, 2), dtype=np.float64)
    essential = {k: np.array(v, dtype=np.float64) for k, v in essential_coords.items()}

    return finite, essential, finite_idx, essential_idx


def wasserstein_matching(dgm_1, dgm_2, q: float = 2.0, delta: float = 0.01, internal_p: float = np.inf,
                        wasserstein_q: typing.Optional[float] = None,
                        ignore_inf_points: bool = True,
                        dim: typing.Optional[int] = None):
    """
    Compute optimal Wasserstein matching between two persistence diagrams.

    Returns a DiagramMatching object containing all matched pairs with
    indices relative to the input diagrams. This allows easy visualization
    of the matching.

    Parameters
    ----------
    dgm_1 : array-like or Diagrams
        First persistence diagram as (n_points, 2) array or Oineus Diagrams object.

    dgm_2 : array-like or Diagrams
        Second persistence diagram.

    q : float, default=2.0
        Wasserstein exponent. Distance = (cost)^(1/q).

    delta : float, default=0.01
        Relative error for approximate matching (Hera auction algorithm).

    internal_p : float, default=np.inf
        Ground metric in the plane. Use np.inf for L_infinity norm.

    wasserstein_q : float, optional
        Alias for q (for API compatibility).

    ignore_inf_points : bool, default=True
        If True, ignore essential points (with infinite coordinates).
        If False, essential points must match by category and cardinality.

    dim : int, optional
        Dimension to extract when dgm_1/dgm_2 are Diagrams objects.

    Returns
    -------
    DiagramMatching
        Object containing all matching information and the distance.

    Raises
    ------
    ValueError
        If essential point cardinalities don't match (when ignore_inf_points=False).

    Examples
    --------
    Basic usage with finite points:

    >>> import numpy as np
    >>> import oineus
    >>>
    >>> dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
    >>> dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])
    >>>
    >>> matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
    >>>
    >>> # Check finite-to-finite matches
    >>> print(matching.finite_to_finite)
    [(0, 0), (1, 1)]
    >>>
    >>> # Verify distance
    >>> print(f"Distance: {matching.distance:.4f}")

    With essential points:

    >>> dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
    >>> dgm_b = np.array([[0.1, 0.9], [0.6, np.inf]])
    >>>
    >>> matching = oineus.wasserstein_matching(dgm_a, dgm_b,
    ...                                        ignore_inf_points=False)
    >>>
    >>> # Essential points matched by sorting finite coordinates
    >>> print(matching.essential_matches)
    {'(finite, +inf)': [(1, 1)]}

    See Also
    --------
    wasserstein_distance : Compute only the distance (no matching info)
    DiagramMatching : Matching result object
    point_to_diagonal : Get diagonal projection coordinates
    """
    # Import here to avoid circular dependency
    from . import _normalize_diagram_for_distance, wasserstein_distance

    # 1. Normalize inputs
    dgm_1 = _normalize_diagram_for_distance(dgm_1, dim=dim)
    dgm_2 = _normalize_diagram_for_distance(dgm_2, dim=dim)

    # 2. Handle wasserstein_q alias
    if wasserstein_q is not None:
        q = wasserstein_q

    # 3. Split into finite and essential with index tracking
    finite_1, essential_1, finite_indices_1, essential_indices_1 = _split_with_indices(dgm_1)
    finite_2, essential_2, finite_indices_2, essential_indices_2 = _split_with_indices(dgm_2)

    # 4. Create result object
    result = DiagramMatching()

    # 5. Match essential points (if not ignored)
    if not ignore_inf_points:
        for category in ["(finite, +inf)", "(finite, -inf)", "(+inf, finite)", "(-inf, finite)"]:
            coords_1 = essential_1[category]
            coords_2 = essential_2[category]
            indices_1 = essential_indices_1[category]
            indices_2 = essential_indices_2[category]

            if len(coords_1) != len(coords_2):
                raise ValueError(
                    f"Essential point cardinalities must match for {category}. "
                    f"Got {len(coords_1)} and {len(coords_2)} points."
                )

            if len(coords_1) > 0:
                # Sort and match by rank
                sorted_1 = np.argsort(coords_1)
                sorted_2 = np.argsort(coords_2)

                matches = [(indices_1[sorted_1[i]], indices_2[sorted_2[i]])
                          for i in range(len(coords_1))]
                result.essential_matches[category] = matches

    # 6. Match finite points using C++
    if len(finite_1) > 0 or len(finite_2) > 0:
        # Convert internal_p to Hera format
        internal_p_hera = -1.0 if np.isinf(internal_p) else internal_p

        # Call C++ matching (returns indices relative to finite arrays)
        cpp_matching = _oineus.wasserstein_matching_finite(
            finite_1, finite_2,
            wasserstein_q=q,
            wasserstein_delta=delta,
            internal_p=internal_p_hera
        )

        # Translate indices back to original diagrams
        result.finite_to_finite = [
            (finite_indices_1[i], finite_indices_2[j])
            for i, j in zip(cpp_matching.a_to_b, cpp_matching.b_from_a)
        ]

        result.a_to_diagonal = [finite_indices_1[i] for i in cpp_matching.a_to_diag]
        result.b_to_diagonal = [finite_indices_2[i] for i in cpp_matching.b_to_diag]

    # 7. Compute distance
    result.distance = wasserstein_distance(dgm_1, dgm_2, q=q, delta=delta,
                                          internal_p=internal_p,
                                          check_for_zero=False)
    result.cost = result.distance ** q

    return result
