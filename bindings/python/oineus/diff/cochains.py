"""Topological optimization with birth and death cochains.

Implements Weighill & Zhou, "Topological optimization with birth and death
cochains" (arXiv:2603.25575). For a bar [b, d) in degree k of a filtration
and a window eps:

- the eps-BIRTH cochain is the unique ell2-minimal real cocycle on
  X_{b+eps} representing the born class and vanishing on X_{b-eps};
- the eps-DEATH cochain is delta(beta_hat), where the death potential
  beta_hat extends the dying cocycle from X_{d-eps} to X_{d+eps} with
  minimal coboundary norm (a Dirichlet / Laplace-learning problem).

Both are computed by sparse least squares over signed (real-coefficient)
coboundary matrices built from the filtration's simplices. Cocycle
representatives come from dionysus persistent cohomology over Z/p (p=47
by default), lifted to integers DREiMac-style; degree 0 uses the
union-find indicator and needs no dionysus.

The CONTENT losses are cochain-weighted averages of filtration values:
with the cochain held fixed (detached), they are linear in the values of
a DiffFiltration, so gradients flow through the existing differentiable
value recomputation to points / pixels / weights. As eps -> 0 the
cochains reduce to the indicator of the single birth/death simplex and
the persistence content converges to d - b.

All snapshot thresholds use the C++ filtration's float64 values, never
the differentiable tensor.
"""

import typing
import warnings
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # torch is only needed by the content losses
    torch = None

from ._backend import require_torch


def _under(fil):
    return getattr(fil, "under_fil", fil)


@dataclass
class _ComplexData:
    """Per-dimension simplex/value arrays of a filtration, in filtration
    order within each dimension (so a sublevel snapshot is a per-dim
    prefix). All within-dim indices refer to this order."""
    simplices: typing.Dict[int, np.ndarray]   # dim -> (n_d, dim+1) ascending vertex rows
    values: typing.Dict[int, np.ndarray]      # dim -> (n_d,) float64, non-decreasing
    dim_first: typing.Dict[int, int]          # dim -> global sorted-id offset
    index_of: typing.Dict[int, dict]          # dim -> {vertex tuple: within-dim index}


def _complex_data(fil, max_needed_dim):
    under = _under(fil)
    if getattr(under, "negate", False):
        raise ValueError("cochains: negate=True filtrations are not supported")
    simplices, values, dim_first, index_of = {}, {}, {}, {}
    offsets = list(under.dim_first)
    for q in range(min(max_needed_dim, under.max_dim) + 1):
        verts = np.sort(np.ascontiguousarray(under.get_simplices_as_arr(q), dtype=np.int64), axis=1)
        n_q = len(verts)
        first = offsets[q]
        vals = np.array([under.cell_value_by_sorted_id(first + i) for i in range(n_q)], dtype=np.float64)
        assert np.all(np.diff(vals) >= 0), "values within a dimension must be non-decreasing"
        simplices[q] = verts
        values[q] = vals
        dim_first[q] = first
        index_of[q] = {tuple(row): i for i, row in enumerate(map(tuple, verts))}
    return _ComplexData(simplices, values, dim_first, index_of)


def _prefix_count(cdata, dim, t):
    if dim not in cdata.values:
        return 0
    return int(np.searchsorted(cdata.values[dim], t, side="right"))


def _signed_coboundary(cdata, k, n_k, n_k1):
    """Signed coboundary delta_k restricted to a snapshot: (n_k1, n_k)
    sparse int8 matrix over the first n_k1 (k+1)-simplices and first n_k
    k-simplices; facet i (drop vertex i of the ascending row) has sign
    (-1)^i. Requires closure: every facet of a snapshot simplex is in the
    snapshot (guaranteed by value monotonicity)."""
    import scipy.sparse as sp

    if n_k1 == 0 or n_k == 0:
        return sp.csr_matrix((n_k1, n_k), dtype=np.int8)
    rows_idx, cols_idx, signs = [], [], []
    top = cdata.simplices[k + 1][:n_k1]
    idx = cdata.index_of[k]
    for r, row in enumerate(map(tuple, top)):
        for i in range(k + 2):
            facet = row[:i] + row[i + 1:]
            c = idx[facet]
            assert c < n_k, "snapshot not closed: facet outside the prefix"
            rows_idx.append(r)
            cols_idx.append(c)
            signs.append(1 if i % 2 == 0 else -1)
    return sp.csr_matrix((np.array(signs, dtype=np.int8), (rows_idx, cols_idx)),
                         shape=(n_k1, n_k))


def _select_bar(fil, dim, bar):
    """Resolve a bar spec to (b, d, birth_sorted_id, death_sorted_id).

    bar: "longest", an int i (the i-th longest bar), or a (b, d) pair
    matched against the diagram with relative tolerance.
    """
    from .. import Decomposition, ReductionParams

    under = _under(fil)
    dcmp = Decomposition(under, False)
    dcmp.reduce(ReductionParams())
    dgms = dcmp.diagram(fil=under, include_inf_points=False)
    v = np.asarray(dgms.in_dimension(dim), dtype=np.float64)
    ix = np.asarray(dgms.index_diagram_in_dimension(dim), dtype=np.int64)
    if len(v) == 0:
        raise ValueError(f"cochains: no finite bars in dimension {dim}")
    pers_order = np.argsort(v[:, 0] - v[:, 1])  # decreasing persistence
    if isinstance(bar, str):
        if bar != "longest":
            raise ValueError(f"cochains: unknown bar spec {bar!r}")
        i = pers_order[0]
    elif isinstance(bar, (int, np.integer)):
        i = pers_order[int(bar)]
    else:
        b, d = bar
        close = np.flatnonzero(np.isclose(v[:, 0], b, rtol=1e-9, atol=1e-12)
                               & np.isclose(v[:, 1], d, rtol=1e-9, atol=1e-12))
        if len(close) != 1:
            raise ValueError(f"cochains: bar {bar} matches {len(close)} diagram points")
        i = close[0]
    return float(v[i, 0]), float(v[i, 1]), int(ix[i, 0]), int(ix[i, 1])


def _vertex_components(cdata, t):
    """Connected-component labels of the vertices of X_t (within-dim-0
    indices), using the edges of X_t."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n_v = _prefix_count(cdata, 0, t)
    n_e = _prefix_count(cdata, 1, t)
    vidx = cdata.index_of[0]
    ii, jj = [], []
    for row in map(tuple, cdata.simplices[1][:n_e]):
        ii.append(vidx[(row[0],)])
        jj.append(vidx[(row[1],)])
    graph = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n_v, n_v))
    return connected_components(graph, directed=False)[1]


@dataclass
class Cochain:
    """A birth or death cochain of one bar.

    sorted_ids are GLOBAL sorted ids of the support simplices (usable to
    index DiffFiltration.values); weights = |coeffs| / ||coeffs||_1 are
    the content gradients w.r.t. the support simplices' values.
    """
    degree: int
    sorted_ids: np.ndarray
    weights: np.ndarray
    coeffs: np.ndarray
    t_K: float
    t_L: float

    def __str__(self):
        return (f"Cochain(degree={self.degree}, support={len(self.sorted_ids)}, "
                f"t_K={self.t_K:.6g}, t_L={self.t_L:.6g})")

    __repr__ = __str__


def _make_cochain(cdata, degree, coeffs, t_K, t_L, tol=1e-10):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    mask = np.abs(coeffs) > tol * max(np.abs(coeffs).max(), 1e-300)
    within = np.flatnonzero(mask)
    c = coeffs[within]
    l1 = np.abs(c).sum()
    if l1 == 0:
        raise RuntimeError("cochains: computed cochain is zero")
    return Cochain(degree=degree,
                   sorted_ids=cdata.dim_first[degree] + within,
                   weights=np.abs(c) / l1,
                   coeffs=c,
                   t_K=t_K, t_L=t_L)


# ---------------------------------------------------------------------------
# cocycle representatives
# ---------------------------------------------------------------------------

def _lifted_cocycle(cdata, dim, b, d, t_valid, prime=47):
    """Integer cocycle representative of the bar [b, d) in degree dim,
    valid on the snapshot X_{t_valid} (t_valid < d).

    Runs dionysus persistent cohomology over Z/prime on the filtration
    truncated at t_valid (dionysus only keeps cocycles of classes alive at
    the end, and truncation keeps our bar alive), matches the bar by birth
    value, lifts coefficients to (-p/2, p/2], and repairs the lift if it
    fails the exact integer cocycle test.
    """
    try:
        import dionysus
    except ImportError as e:
        raise ImportError("cochains: degree >= 1 cocycle representatives need the dionysus "
                          "package (dev dependency); pip install dionysus") from e

    f = dionysus.Filtration()
    for q in range(min(dim + 1, max(cdata.simplices)) + 1):
        n_q = _prefix_count(cdata, q, t_valid)
        for row, val in zip(cdata.simplices[q][:n_q], cdata.values[q][:n_q]):
            f.append(dionysus.Simplex([int(v) for v in row], float(val)))
    f.sort()
    pers = dionysus.cohomology_persistence(f, prime, True)
    dgms = dionysus.init_diagrams(pers, f)

    candidates = [pt for pt in dgms[dim]
                  if pt.death == float("inf") and np.isclose(pt.birth, b, rtol=1e-5, atol=1e-7)]
    if len(candidates) != 1:
        raise RuntimeError(f"cochains: bar (b={b:.6g}, d={d:.6g}) matched {len(candidates)} "
                           f"alive classes in the truncated filtration (dionysus stores float32 "
                           f"values; perturb the input if two bars collide within 1e-5)")
    point = candidates[0]

    n_k = _prefix_count(cdata, dim, t_valid)
    alpha = np.zeros(n_k, dtype=np.int64)
    idx = cdata.index_of[dim]
    for entry in pers.cocycle(point.data):
        simplex = f[entry.index]
        key = tuple(sorted(v for v in simplex))
        e = int(entry.element)
        alpha[idx[key]] = e if e <= prime // 2 else e - prime

    # exact integer cocycle test on X_{t_valid}
    n_k1 = _prefix_count(cdata, dim + 1, t_valid)
    delta = _signed_coboundary(cdata, dim, n_k, n_k1).astype(np.int64)
    residual = delta @ alpha
    if np.any(residual):
        alpha = _fix_integer_lift(alpha, residual, delta, prime)
    return alpha


def _fix_integer_lift(alpha, residual, delta, prime):
    """Repair an integer lift that is a cocycle mod p but not over Z:
    find integer y with delta y = residual / p and subtract p*y (mirrors
    fix_integer_lift in the reference implementation)."""
    from scipy.sparse.linalg import lsqr

    if np.any(residual % prime):
        raise RuntimeError("cochains: lifted cochain is not a cocycle mod p; "
                           "inconsistent coboundary sign convention?")
    z = residual // prime
    y_real = lsqr(delta.astype(np.float64), z.astype(np.float64), atol=1e-14, btol=1e-14)[0]
    y = np.rint(y_real).astype(np.int64)
    if np.array_equal(delta @ y, z):
        return alpha - prime * y
    # exact integer solve via MILP (rare: only when the rounded lsqr fails)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        res = milp(c=np.zeros(delta.shape[1]),
                   constraints=LinearConstraint(delta.astype(np.float64), z, z),
                   integrality=np.ones(delta.shape[1]),
                   bounds=Bounds(-prime, prime))
        if res.success:
            y = np.rint(res.x).astype(np.int64)
            if np.array_equal(delta @ y, z):
                return alpha - prime * y
    except Exception:
        pass
    warnings.warn("cochains: could not repair the integer cocycle lift exactly; "
                  "falling back to a real least-squares correction")
    return alpha - prime * y_real


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------

def _resolve_eps(b, d, eps, relative_eps):
    # the bound is (d - b)/2, not (d - b): the shared cocycle representative
    # is only valid on X_{d - min_eps}, and the birth snapshot X_{b + eps}
    # must stay inside it
    eps_abs = float(eps) * (d - b) if relative_eps else float(eps)
    if not 0 < eps_abs < (d - b) / 2:
        raise ValueError(f"cochains: eps must give 0 < eps_abs < (d - b)/2, got eps_abs={eps_abs} "
                         f"for the bar ({b:.6g}, {d:.6g})")
    return eps_abs


def _check_genericity(cdata, dims, thresholds, rtol=1e-9):
    for q in dims:
        if q not in cdata.values:
            continue
        vals = cdata.values[q]
        for t in thresholds:
            if np.any(np.abs(vals - t) <= rtol * max(abs(t), 1.0)):
                warnings.warn(f"cochains: a dimension-{q} filtration value coincides with a "
                              f"snapshot threshold {t:.6g}; the cochain (and its gradient) is "
                              f"not well-defined at this configuration")


def _zero_pad(vec, n):
    out = np.zeros(n, dtype=np.float64)
    out[:len(vec)] = vec
    return out


def _birth_cochain_coeffs(cdata, dim, b, d, eps_abs, prime, cocycle):
    """ell2-minimal real cocycle on L = X_{b+eps} in the born class,
    vanishing on K = X_{b-eps}. Returns coefficients over L's k-simplices."""
    from scipy.sparse.linalg import lsqr

    t_K, t_L = b - eps_abs, b + eps_abs
    if dim == 0:
        # paper, degree 0: indicator of the vertices that merge with the
        # birth vertex before d, restricted to X_{b+eps}
        comp = _vertex_components(cdata, d - eps_abs)
        birth_within = cocycle  # for dim 0, `cocycle` carries the birth vertex within-idx
        n_v_L = _prefix_count(cdata, 0, t_L)
        alpha = (comp[:n_v_L] == comp[birth_within]).astype(np.float64)
        return alpha, t_K, t_L
    if dim >= 2:
        raise NotImplementedError("cochains: birth cochains are implemented for degrees 0 and 1")

    n_k_K, n_k_L = _prefix_count(cdata, dim, t_K), _prefix_count(cdata, dim, t_L)
    n_km1_L = _prefix_count(cdata, dim - 1, t_L)
    n_km1_K = _prefix_count(cdata, dim - 1, t_K)

    alpha = np.asarray(cocycle[:n_k_L], dtype=np.float64).copy()

    # step 1: make alpha|_K an exact zero by subtracting a coboundary
    # (the class dies under restriction to K, so alpha|_K is a coboundary)
    delta_K = _signed_coboundary(cdata, dim - 1, n_km1_K, n_k_K).astype(np.float64)
    delta_L = _signed_coboundary(cdata, dim - 1, n_km1_L, n_k_L).astype(np.float64)
    if n_k_K > 0:
        g = lsqr(delta_K, alpha[:n_k_K], atol=1e-14, btol=1e-14)[0]
        alpha -= delta_L @ _zero_pad(g, n_km1_L)
        if np.abs(alpha[:n_k_K]).max(initial=0.0) > 1e-6:
            raise RuntimeError("cochains: restriction of the cocycle to X_{b-eps} is not a "
                               "coboundary; wrong bar or broken representative")
        alpha[:n_k_K] = 0.0

    # step 2: minimize the ell2 norm over corrections delta(h) that keep
    # alpha zero on K: h free on new vertices, locally constant on K
    import scipy.sparse as sp
    comp_K = _vertex_components(cdata, t_K) if n_km1_K > 0 else np.zeros(0, dtype=int)
    n_comp = comp_K.max() + 1 if len(comp_K) else 0
    n_new = n_km1_L - n_km1_K
    cols = []
    if n_comp:
        indicator = sp.coo_matrix((np.ones(n_km1_K), (np.arange(n_km1_K), comp_K)),
                                  shape=(n_km1_L, n_comp))
        cols.append(indicator)
    if n_new:
        eye_new = sp.coo_matrix((np.ones(n_new), (np.arange(n_km1_K, n_km1_L), np.arange(n_new))),
                                shape=(n_km1_L, n_new))
        cols.append(eye_new)
    if cols:
        G = sp.hstack(cols).tocsr()
        M = (delta_L @ G).tocsr()
        c = lsqr(M, -alpha, atol=1e-14, btol=1e-14)[0]
        alpha = alpha + M @ c
        alpha[:n_k_K] = 0.0
    return alpha, t_K, t_L


def _death_cochain_coeffs(cdata, dim, d, eps_abs, beta_K):
    """Death cochain: extend the dying cocycle beta from K = X_{d-eps} to
    L = X_{d+eps} with minimal coboundary norm; the cochain is the
    coboundary of that extension, supported on the new (k+1)-simplices.
    Returns coefficients over L's (k+1)-simplices."""
    from scipy.sparse.linalg import lsqr

    t_K, t_L = d - eps_abs, d + eps_abs
    n_k_K, n_k_L = _prefix_count(cdata, dim, t_K), _prefix_count(cdata, dim, t_L)
    n_k1_K, n_k1_L = _prefix_count(cdata, dim + 1, t_K), _prefix_count(cdata, dim + 1, t_L)

    delta_L = _signed_coboundary(cdata, dim, n_k_L, n_k1_L).astype(np.float64)
    beta_ext = _zero_pad(np.asarray(beta_K[:n_k_K], dtype=np.float64), n_k_L)
    rhs = -(delta_L @ beta_ext)
    new_cols = np.arange(n_k_K, n_k_L)
    if len(new_cols):
        x = lsqr(delta_L[:, new_cols], rhs, atol=1e-14, btol=1e-14)[0]
        beta_ext[new_cols] = x
    omega = delta_L @ beta_ext
    if n_k1_K and np.abs(omega[:n_k1_K]).max(initial=0.0) > 1e-8:
        raise RuntimeError("cochains: death cochain does not vanish on X_{d-eps}; "
                           "beta is not a cocycle there")
    omega[:n_k1_K] = 0.0
    return omega, t_K, t_L


def _bar_and_cocycle(fil, dim, bar, eps_list, relative_eps, prime):
    """Shared setup: select the bar, resolve eps values, fetch one cocycle
    representative valid on X_{d - min_eps} (restrictions to the smaller
    snapshots are automatic)."""
    cdata = _complex_data(fil, dim + 1)
    b, d, b_sid, d_sid = _select_bar(fil, dim, bar)
    eps_abs = [_resolve_eps(b, d, e, relative_eps) for e in eps_list]
    thresholds = [t for e in eps_abs for t in (b - e, b + e, d - e, d + e)]
    _check_genericity(cdata, range(dim + 2), thresholds)
    t_valid = d - min(eps_abs)
    if dim == 0:
        # within-dim-0 index of the birth vertex; the union-find indicator
        # replaces the lifted cocycle
        birth_within = b_sid - cdata.dim_first[0]
        cocycle = birth_within
    else:
        cocycle = _lifted_cocycle(cdata, dim, b, d, t_valid, prime)
    return cdata, b, d, eps_abs, cocycle


def birth_cochain(fil, dim=1, bar="longest", eps=0.05, *, relative_eps=True, prime=47):
    """The eps-birth cochain of a bar as a Cochain (degree = dim)."""
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, dim, bar, [eps], relative_eps, prime)
    alpha, t_K, t_L = _birth_cochain_coeffs(cdata, dim, b, d, eps_abs[0], prime, cocycle)
    return _make_cochain(cdata, dim, alpha, t_K, t_L)


def death_cochain(fil, dim=1, bar="longest", eps=0.05, *, relative_eps=True, prime=47):
    """The eps-death cochain of a bar as a Cochain (degree = dim + 1)."""
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, dim, bar, [eps], relative_eps, prime)
    beta = _beta_from_cocycle(cdata, dim, d, eps_abs[0], cocycle)
    omega, t_K, t_L = _death_cochain_coeffs(cdata, dim, d, eps_abs[0], beta)
    return _make_cochain(cdata, dim + 1, omega, t_K, t_L)


def birth_death_cochains(fil, dim=1, bar="longest", eps=0.05, *, relative_eps=True, prime=47):
    """Both eps-cochains of a bar, sharing one cocycle representative."""
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, dim, bar, [eps], relative_eps, prime)
    alpha, tb_K, tb_L = _birth_cochain_coeffs(cdata, dim, b, d, eps_abs[0], prime, cocycle)
    beta = _beta_from_cocycle(cdata, dim, d, eps_abs[0], cocycle)
    omega, td_K, td_L = _death_cochain_coeffs(cdata, dim, d, eps_abs[0], beta)
    return (_make_cochain(cdata, dim, alpha, tb_K, tb_L),
            _make_cochain(cdata, dim + 1, omega, td_K, td_L))


def _beta_from_cocycle(cdata, dim, d, eps_abs, cocycle):
    """Dying-class representative on X_{d-eps}: restriction of the lifted
    cocycle (degree >= 1) or the component indicator (degree 0). The death
    cochain is representative-independent (Cor 4.4 of the paper)."""
    n_k_K = _prefix_count(cdata, dim, d - eps_abs)
    if dim == 0:
        comp = _vertex_components(cdata, d - eps_abs)
        return (comp[:n_k_K] == comp[cocycle]).astype(np.float64)
    return np.asarray(cocycle[:n_k_K], dtype=np.float64)


# ---------------------------------------------------------------------------
# differentiable content losses
# ---------------------------------------------------------------------------

def _content_terms(fil, cochain, cdata, edge_relaxed):
    """Torch scalar: content of one cochain = sum of weights * values.

    With edge_relaxed (degree-2 cochains on VR-like filtrations), each
    support simplex's weight is split equally over its NEW facets (facets
    with value > t_K), following the paper's edge-relaxed death content --
    this keeps the loss meaningful for VR, where moving a triangle's value
    means moving its longest edge.
    """
    values = fil.values
    if not edge_relaxed:
        ids = torch.from_numpy(np.ascontiguousarray(cochain.sorted_ids))
        w = torch.from_numpy(np.ascontiguousarray(cochain.weights)).to(values.dtype)
        return (w * values[ids]).sum()

    k1 = cochain.degree
    k = k1 - 1
    n_k_K = _prefix_count(cdata, k, cochain.t_K)
    idx = cdata.index_of[k]
    edge_weight = {}
    for sid, w in zip(cochain.sorted_ids, cochain.weights):
        row = tuple(cdata.simplices[k1][sid - cdata.dim_first[k1]])
        facets = [row[:i] + row[i + 1:] for i in range(k1 + 1)]
        new_facets = [ftup for ftup in facets if idx[ftup] >= n_k_K]
        if not new_facets:
            raise RuntimeError("cochains: edge_relaxed needs every support simplex to have a "
                               "new facet in the window (holds for VR); pass edge_relaxed=False")
        for ftup in new_facets:
            gid = cdata.dim_first[k] + idx[ftup]
            edge_weight[gid] = edge_weight.get(gid, 0.0) + w / len(new_facets)
    ids = torch.from_numpy(np.fromiter(edge_weight.keys(), dtype=np.int64))
    w = torch.from_numpy(np.fromiter(edge_weight.values(), dtype=np.float64)).to(values.dtype)
    return (w * values[ids]).sum()


def birth_content(fil, dim=1, bar="longest", eps=0.05, *, relative_eps=True, prime=47):
    """Differentiable eps-birth content B_eps of a bar (torch scalar)."""
    require_torch(fil.values, "birth_content")
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, dim, bar, [eps], relative_eps, prime)
    alpha, t_K, t_L = _birth_cochain_coeffs(cdata, dim, b, d, eps_abs[0], prime, cocycle)
    cochain = _make_cochain(cdata, dim, alpha, t_K, t_L)
    return _content_terms(fil, cochain, cdata, edge_relaxed=False)


def _resolve_edge_relaxed(fil, edge_relaxed, dim):
    # edge-relaxed content assumes every death support simplex has a facet
    # appearing in the window, which holds for VR simplices of dim >= 2 (a
    # VR simplex enters with its longest edge, and some facet contains it)
    # but not for Cech/alpha-style MEB values, nor for degree-1 death
    # cochains (edges; their facets are vertices, all old); None
    # auto-selects by the filtration kind and degree
    if edge_relaxed is not None:
        return edge_relaxed
    from .. import _oineus
    return dim >= 1 and _under(fil).kind == _oineus.FiltrationKind.Vr


def death_content(fil, dim=1, bar="longest", eps=0.05, *, relative_eps=True,
                  edge_relaxed=None, prime=47):
    """Differentiable eps-death content D_eps of a bar (torch scalar).

    edge_relaxed: split each support simplex's weight over its NEW facets
    (the paper's edge-relaxed variant); None (default) enables it exactly
    for VR filtrations, where it is well-defined.
    """
    require_torch(fil.values, "death_content")
    edge_relaxed = _resolve_edge_relaxed(fil, edge_relaxed, dim)
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, dim, bar, [eps], relative_eps, prime)
    beta = _beta_from_cocycle(cdata, dim, d, eps_abs[0], cocycle)
    omega, t_K, t_L = _death_cochain_coeffs(cdata, dim, d, eps_abs[0], beta)
    cochain = _make_cochain(cdata, dim + 1, omega, t_K, t_L)
    return _content_terms(fil, cochain, cdata, edge_relaxed)


def persistence_content_loss(fil, dim=1, bar="longest", eps=(0.01, 0.05, 0.1), *,
                             relative_eps=True, mode="persistence",
                             edge_relaxed=None, prime=47):
    """Differentiable multi-eps content of one bar (torch scalar).

    mode: "persistence" -> mean over eps of (D_eps - B_eps) (converges to
    d - b as eps -> 0; MAXIMIZE it to grow the bar); "birth" -> mean
    B_eps; "death" -> mean D_eps. One cocycle representative is shared
    across all eps values. edge_relaxed applies to the death cochain only;
    None (default) enables it exactly for VR filtrations.
    """
    require_torch(fil.values, "persistence_content_loss")
    edge_relaxed = _resolve_edge_relaxed(fil, edge_relaxed, dim)
    if mode not in ("persistence", "birth", "death"):
        raise ValueError(f"cochains: unknown mode {mode!r}")
    if np.isscalar(eps):
        eps = (eps,)
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, dim, bar, list(eps), relative_eps, prime)
    terms = []
    for e in eps_abs:
        if mode in ("birth", "persistence"):
            alpha, t_K, t_L = _birth_cochain_coeffs(cdata, dim, b, d, e, prime, cocycle)
            b_term = _content_terms(fil, _make_cochain(cdata, dim, alpha, t_K, t_L),
                                    cdata, edge_relaxed=False)
        if mode in ("death", "persistence"):
            beta = _beta_from_cocycle(cdata, dim, d, e, cocycle)
            omega, t_K, t_L = _death_cochain_coeffs(cdata, dim, d, e, beta)
            d_term = _content_terms(fil, _make_cochain(cdata, dim + 1, omega, t_K, t_L),
                                    cdata, edge_relaxed)
        if mode == "persistence":
            terms.append(d_term - b_term)
        elif mode == "birth":
            terms.append(b_term)
        else:
            terms.append(d_term)
    return sum(terms) / len(terms)
