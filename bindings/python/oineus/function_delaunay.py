"""Two-value function-Delaunay (Cech-Delaunay) bifiltration.

Implements the incremental Delaunay complex I(X) of Alonso, Kerber, Lam,
Lesnick, "Delaunay Bifiltrations of Functions on Point Clouds" (SODA 2024,
arXiv:2310.15902), specialized to a function gamma with two values: points
of group A (gamma = 1) ordered before points of group B (gamma = 2).

I(X) is monotone under point insertion (unlike the Delaunay triangulation
itself), so the slice of all-A simplices, I(A), is a genuine subcomplex of
I(X) -- exactly the nested pair L subseteq K that kernel/image/cokernel
persistence and mixup barcodes need. Each simplex carries its squared MEB
radius (the Delaunay-Cech convention), so by the paper's collapse theorem
the L slice has the persistence diagrams of the union of balls around A,
and the K slice those of the union of balls around all points.

Construction: with the Delta-property enforced by d+1 far dummy points,
every simplex of I(X) is a face of a Bowyer-Watson conflict (d+1)-simplex
(purity lemma), so it suffices to record the conflict simplices of every
insertion, take one downward closure, and strip dummy-touching simplices.
"""

import typing

import numpy as np


def _enclosing_dummy_simplex(points, dummy_scale):
    """Regular d-simplex of d+1 far points strictly enclosing the cloud.

    Enforces the Delta-property (purity of I(X)) and guarantees every
    real insertion point is inside the current hull, so conflict location
    never walks off a hull facet.
    """
    d = points.shape[1]
    centroid = points.mean(axis=0)
    diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    diameter = max(diameter, 1.0)
    radius = dummy_scale * diameter
    if d == 2:
        angles = np.array([0.5, 0.5 + 2 / 3, 0.5 + 4 / 3]) * np.pi
        simplex = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        simplex = radius / np.sqrt(3.0) * np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    simplex = simplex + centroid
    # sufficient strict-enclosure check: the insphere of a regular d-simplex
    # with circumradius R has radius R/d
    max_dist = float(np.max(np.linalg.norm(points - centroid, axis=1)))
    if max_dist >= radius / d:
        raise ValueError("function_delaunay_bifiltration: dummy_scale too small to enclose the data")
    return simplex


def _circumsphere_of(pts):
    # circumcenter offset y = c - pts[0] and squared radius of a
    # full-dimensional simplex, via the linearized equidistance system;
    # a singular system (degenerate sliver) reports r_sq = inf
    rel = pts[1:] - pts[0]
    rhs = 0.5 * np.sum(rel * rel, axis=1)
    try:
        y = np.linalg.solve(rel, rhs)
    except np.linalg.LinAlgError:
        return None, np.inf
    return y, float(np.sum(y * y))


def _conflicts_bfs(tri, p, eps):
    """Indices of the d-simplices of tri whose circumsphere strictly
    contains p (the Bowyer-Watson cavity), found by BFS from the simplex
    containing p. The cavity is connected and contains that simplex.

    Two-tier predicate: a simplex is RECORDED as a conflict only when p is
    strictly inside its circumsphere (relative margin eps), but the flood
    fill TRAVERSES also near-ties and degenerate slivers -- otherwise one
    mis-classified simplex inside the cavity would disconnect the search
    and silently drop conflicts behind it. The traversal shell beyond the
    true cavity is one layer of near-ties, so the cost stays local.
    """
    s0 = int(tri.find_simplex(p.reshape(1, -1))[0])
    if s0 == -1:
        raise RuntimeError("function_delaunay_bifiltration: insertion point outside the hull "
                           "(dummy simplex failed to enclose the data)")
    eps_relax = max(1e-6, eps)
    conflicts = []
    visited = {s0}
    stack = [s0]
    pts = tri.points
    while stack:
        s = stack.pop()
        verts = tri.simplices[s]
        y, r_sq = _circumsphere_of(pts[verts])
        if np.isinf(r_sq):
            record, traverse = False, True
        else:
            dist_sq = float(np.sum((p - pts[verts[0]] - y) ** 2))
            record = dist_sq < r_sq * (1 - eps)
            traverse = dist_sq < r_sq * (1 + eps_relax)
        if record:
            conflicts.append(s)
        if traverse:
            for nb in tri.neighbors[s]:
                if nb != -1 and nb not in visited:
                    visited.add(int(nb))
                    stack.append(int(nb))
    return conflicts


def _simplex_rows(tri):
    return {tuple(sorted(int(v) for v in row)) for row in tri.simplices}


def function_delaunay_bifiltration(points, labels, *,
                                   a_value=None,
                                   max_dim: int = -1,
                                   squared: bool = True,
                                   jitter: float = 0.0,
                                   seed: typing.Optional[int] = None,
                                   dummy_scale: float = 20.0,
                                   conflict_method: str = "bfs",
                                   eps: float = 1e-10,
                                   n_threads: int = 1):
    """Build the two-slice function-Delaunay bifiltration (K, L).

    Points of group A (labels == a_value) get function value 1, the rest
    (group B) get 2. Returns the pair of nested filtrations

        L = I(A)  subseteq  K = I(all points),

    where I is the incremental Delaunay complex of Alonso-Kerber-Lam-
    Lesnick and every simplex carries the (squared) radius of the minimum
    enclosing ball of its vertices. L is built with K.subfiltration, so
    shared cells have identical uids and values -- feed the pair directly
    to compute_kernel_image_cokernel_reduction or
    mixup_barcodes_of_filtrations.

    Persistence guarantees (collapse theorem of the paper): diagrams of L
    equal the alpha/Cech diagrams of the A points alone, diagrams of K
    those of the full cloud; the kernel/image/cokernel diagrams of the
    inclusion are those of union-of-balls(A) -> union-of-balls(all).

    The complex is the max_dim-skeleton of I(X) (I(X) itself reaches
    dimension d+1, whose MEBs would need d+3 support points). Skeleton
    truncation is exact for diagrams and KICR in dimensions
    0 .. max_dim-1; the dimension-max_dim diagram is unreliable.

    Args:
        points: (n, d) array, d in {2, 3}, pairwise-distinct positions.
        labels: (n,) array with exactly two distinct values (the two
            atom species).
        a_value: The label that forms group A (function value 1, the
            subcomplex L). Default: min of the two label values.
        max_dim: Largest simplex dimension to keep; default d. Must be
            <= d (higher-dimensional MEBs are not implemented).
        squared: Use squared MEB radii (the oineus alpha convention,
            default) or plain radii.
        jitter: If positive, add Gaussian noise with this std (in units
            of the median nearest-neighbor distance) to a working COPY of
            the coordinates, used for BOTH the triangulation and the MEB
            values. Required for degenerate inputs (crystal lattices are
            heavily cospherical).
        seed: Seed for the jitter and for the insertion-order shuffle
            within each group. Fixed seed -> identical (K, L).
        dummy_scale: Circumradius of the enclosing dummy simplex as a
            multiple of the data diameter.
        conflict_method: "bfs" (output-sensitive, default) or "setdiff"
            (snapshot difference; slow oracle for small-input tests).
        eps: Relative tolerance of the strict in-circumsphere predicate
            (bfs method).
        n_threads: Threads for the Filtration constructor sort.

    Returns:
        (K, L): oineus Filtrations; L is a subfiltration of K containing
        exactly the simplices whose vertices are all in group A. Vertex
        ids are indices into the input points array.
    """
    from . import REAL_MODULES, _tetrahedron_meb_sq_np, _triangle_meb_sq_np, detect_real_dtype

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("function_delaunay_bifiltration: points must be a 2D array with at least 2 rows")
    n, d = points.shape
    if d not in (2, 3):
        raise ValueError(f"function_delaunay_bifiltration: ambient dimension must be 2 or 3, got {d}")
    labels = np.asarray(labels)
    if labels.shape != (n,):
        raise ValueError("function_delaunay_bifiltration: labels must have shape (n_points,)")
    label_values = np.unique(labels)
    if len(label_values) != 2:
        raise ValueError(f"function_delaunay_bifiltration: labels must take exactly two distinct "
                         f"values, got {len(label_values)}")
    if conflict_method not in ("bfs", "setdiff"):
        raise ValueError(f"function_delaunay_bifiltration: unknown conflict_method {conflict_method!r}")
    if max_dim < 0:
        max_dim = d
    if max_dim > d:
        raise ValueError(f"function_delaunay_bifiltration: max_dim={max_dim} > ambient dimension {d} "
                         "is not supported")
    if a_value is None:
        a_value = label_values[0]
    is_a = labels == a_value
    if not is_a.any():
        raise ValueError("function_delaunay_bifiltration: a_value matches no label")

    rng = np.random.default_rng(seed)
    pts_work = np.ascontiguousarray(points, dtype=np.float64)
    if jitter > 0:
        from scipy.spatial import cKDTree
        nn_dist = cKDTree(pts_work).query(pts_work, k=2)[0][:, 1]
        pts_work = pts_work + jitter * np.median(nn_dist) * rng.standard_normal(pts_work.shape)

    # gamma-order: group A first, then B; the order within each group is a
    # gamma tie-break, so shuffle it (randomized incremental construction
    # keeps |I(X)| near its expectation; sorted orders inflate it)
    order_a = np.flatnonzero(is_a)
    order_b = np.flatnonzero(~is_a)
    rng.shuffle(order_a)
    rng.shuffle(order_b)
    order = np.concatenate([order_a, order_b])

    dummies = _enclosing_dummy_simplex(pts_work, dummy_scale)
    n_dummies = d + 1
    # extended index space: dummies 0..d, then insertion position d+1+j;
    # ext_to_orig maps extended indices to original point indices
    ext_to_orig = np.concatenate([np.full(n_dummies, -1, dtype=np.int64), order])
    pts_ext = np.vstack([dummies, pts_work[order]])

    from scipy.spatial import Delaunay

    # Qhull needs d+2 points to seed, so the first real point goes into the
    # seed triangulation; its conflict simplex would be all-dummy-touching
    # (stripped below) and its singleton is added explicitly, so nothing is
    # lost by not recording it
    tri = Delaunay(pts_ext[:n_dummies + 1], incremental=True)
    top_simplices = set()
    for j in range(1, n):
        ext_idx = n_dummies + j
        p = pts_ext[ext_idx]
        if conflict_method == "bfs":
            for s in _conflicts_bfs(tri, p, eps):
                tau = tuple(sorted(int(v) for v in tri.simplices[s]))
                top_simplices.add(tuple(sorted(tau + (ext_idx,))))
        else:
            before = _simplex_rows(tri)
            tri.add_points(pts_ext[ext_idx:ext_idx + 1])
            for tau in before - _simplex_rows(tri):
                top_simplices.add(tuple(sorted(tau + (ext_idx,))))
            continue
        tri.add_points(pts_ext[ext_idx:ext_idx + 1])
    tri.close()

    # downward closure FIRST (a dummy-touching conflict simplex has real
    # faces that belong to I(X)), then strip dummy-touching simplices;
    # faces above max_dim are dropped (skeleton truncation)
    from itertools import combinations
    real_simplices = set()
    for top in top_simplices:
        real = tuple(v for v in top if v >= n_dummies)
        for k in range(1, min(len(real), max_dim + 1) + 1):
            for face in combinations(real, k):
                real_simplices.add(face)
    # every real point is a 0-cell even if a predicate edge case dropped
    # all of its conflicts
    for ext_idx in range(n_dummies, n_dummies + n):
        real_simplices.add((ext_idx,))

    by_dim = {}
    for simplex in real_simplices:
        by_dim.setdefault(len(simplex) - 1, []).append(simplex)

    dt = detect_real_dtype(points)
    sub = REAL_MODULES[dt]
    verts_by_dim = []
    vals_by_dim = []
    for q in sorted(by_dim):
        if q != len(verts_by_dim):
            raise RuntimeError(f"function_delaunay_bifiltration: no simplices in dimension {len(verts_by_dim)}")
        ext_rows = np.array(sorted(by_dim[q]), dtype=np.int64)
        # MEB values from the (jittered) working coordinates, in the
        # EXTENDED order; vertex relabeling below does not change values
        coords = [pts_ext[ext_rows[:, k]] for k in range(q + 1)]
        if q == 0:
            vals = np.zeros(len(ext_rows), dtype=np.float64)
        elif q == 1:
            vals = 0.25 * np.sum((coords[0] - coords[1]) ** 2, axis=1)
        elif q == 2:
            vals = _triangle_meb_sq_np(coords[0], coords[1], coords[2])
        elif q == 3:
            vals = _tetrahedron_meb_sq_np(coords[0], coords[1], coords[2], coords[3])
        else:
            raise RuntimeError(f"function_delaunay_bifiltration: unexpected simplex dimension {q}")
        if not squared:
            vals = np.sqrt(vals)
        orig_rows = ext_to_orig[ext_rows]
        verts_by_dim.append(np.ascontiguousarray(orig_rows))
        vals_by_dim.append(np.ascontiguousarray(vals, dtype=dt))

    K = sub._filtration_from_arrays(verts_by_dim, vals_by_dim, n_threads=n_threads)
    is_a_list = is_a.tolist()
    L = K.subfiltration(lambda s: all(is_a_list[v] for v in s.vertices))
    return K, L
