"""Framework-neutral core of the differentiable persistence diagram.

numpy in / numpy out. Holds all the 4-matrix bookkeeping that the custom
backward needs, so the torch (pd_torch) and jax (pd_jax) adapters are thin
marshaling shims. Nothing here imports torch or jax.

The forward (pd_forward) reduces one side of the decomposition and collects
the per-dimension integer index diagrams; the backward (pd_backward) turns
an upstream diagram gradient into a gradient on the filtration values,
either by a plain scatter (dgm-loss) or through the critical-set machinery
(crit-sets). See persistence_diagram.py for the user-facing description of
the two gradient methods.

kicr_forward at the bottom is the analogous framework-neutral forward for
kernel/image/cokernel diagrams (see kicr.py); its index diagrams are pure
gather maps into K's values, so no custom backward is needed there.
"""

from dataclasses import dataclass

import numpy as np

from .. import _oineus
from ._reduction_policy import default_dualize_for_filtration
from .top_optimizer import TopologyOptimizer


_STRATEGY_MAP = {
    "avg": _oineus.ConflictStrategy.Avg,
    "max": _oineus.ConflictStrategy.Max,
    "sum": _oineus.ConflictStrategy.Sum,
    "fca": _oineus.ConflictStrategy.FixCritAvg,
}


_U_STRATEGY_MAP = {
    "auto":           _oineus.UStrategy.Auto,
    "row_partial":    _oineus.UStrategy.RowPartial,
    "legacy_in_band": _oineus.UStrategy.LegacyInBand,
}


def resolve_strategy(strategy):
    """Map "avg"/"max"/"sum"/"fca" (or a ConflictStrategy) to the enum."""
    if isinstance(strategy, _oineus.ConflictStrategy):
        return strategy
    try:
        return _STRATEGY_MAP[strategy.lower()]
    except (AttributeError, KeyError):
        raise ValueError(
            f"unknown conflict_strategy {strategy!r}; expected one of "
            f"{sorted(_STRATEGY_MAP)} or an _oineus.ConflictStrategy"
        )


def resolve_u_strategy(u_strategy):
    """Map None/"auto"/"row_partial"/"legacy_in_band" (or a UStrategy) to the enum."""
    if u_strategy is None:
        return _oineus.UStrategy.Auto
    if isinstance(u_strategy, _oineus.UStrategy):
        return u_strategy
    try:
        return _U_STRATEGY_MAP[u_strategy.lower()]
    except (AttributeError, KeyError):
        raise ValueError(
            f"unknown u_strategy {u_strategy!r}; expected one of "
            f"{sorted(_U_STRATEGY_MAP)} or an _oineus.UStrategy"
        )


def determine_needed_matrices(grad_np, negate):
    """Return ``(v_hom, u_hom, v_coh, u_coh)``: which of the four matrices
    the crit-sets backward needs given a diagram gradient (numpy ``(n, 2)``).

    A positive sign on birth/death pushes the value down (we minimize the
    loss), a negative sign pushes it up. Mapping the sign to the move
    direction and the move direction to the matrix (paper:
    decrease_birth -> U_coh, increase_birth -> V_coh,
    increase_death -> U_hom, decrease_death -> V_hom) gives the layout
    below. ``negate=True`` flips value-direction vs filtration-direction,
    so the matrix assignment swaps accordingly.
    """
    if grad_np.size == 0:
        return False, False, False, False
    mx = grad_np.max(axis=0)
    mn = grad_np.min(axis=0)
    flags = [bool(mx[0] > 0), bool(mx[1] > 0), bool(mn[0] < 0), bool(mn[1] < 0)]
    if negate:
        v_coh, u_hom, u_coh, v_hom = flags
    else:
        u_coh, v_hom, v_coh, u_hom = flags
    return v_hom, u_hom, v_coh, u_coh


def select_u_moves(idx, cur, tgt, move, *, side, negate):
    """Pick the rows + bounds the U-side walker on ``side`` will read.

    Filters moves to those in the U-needing direction:
      hom side, non-negate: increase_death (tgt > cur).
      coh side, non-negate: decrease_birth (tgt < cur).
      negate flips both.
    All inputs are 1-D numpy arrays. Returns (rows_fil_idx_list, bounds_list)
    ready for ensure_has_u_hom / ensure_has_u_coh.
    """
    if side == "hom":
        u_dir_mask = (tgt < cur) if negate else (tgt > cur)
    else:
        u_dir_mask = (tgt > cur) if negate else (tgt < cur)

    sel = move & u_dir_mask
    if not bool(sel.any()):
        return [], []
    rows = idx[sel].astype(np.uintp).tolist()
    bounds = tgt[sel].tolist()
    return rows, bounds


@dataclass
class PDForward:
    """Result of the forward reduction: everything the backward needs.

    Holds the live TopologyOptimizer, a detached numpy copy of the
    filtration values, the per-dim integer index diagrams, and the resolved
    reduction/backward parameters. No framework tensors -- the torch adapter
    stows this on the autograd ctx, the jax adapter closes over it (and its
    crit-sets backward re-reduces from scratch instead of reusing top_opt).
    """
    top_opt: object
    values_np: np.ndarray     # (n,) detached filtration values, sorted order
    index_dgm: dict           # dim -> (n_d, 2) int64 [birth_sid, death_sid]
    negate: bool
    dualize: bool
    method: str               # "dgm-loss" | "crit-sets"
    step_size: float
    strategy: object          # _oineus.ConflictStrategy
    max_dim: int
    dims_to_backprop: list
    n_threads: int
    u_strategy: object        # _oineus.UStrategy


def _forward_reduce(top_opt, dualize):
    """Reduce the chosen side with the recipe baked into the optimizer
    (recipe was decided at construction time)."""
    if dualize:
        top_opt.ensure_coh_reduced()
        return top_opt.cohomology_decomposition_ref()
    top_opt.ensure_hom_reduced()
    return top_opt.homology_decomposition_ref()


def pd_forward(under_fil, values_np, *, dualize, method, dims_to_backprop,
               n_threads, u_strategy, conflict_strategy, step_size, max_dim):
    """Reduce one side of the decomposition; return a PDForward.

    under_fil is the C++ filtration; values_np a detached numpy copy of the
    differentiable filtration values (the backward shapes its gradient like
    it and the crit-sets math reads it). Parameters may be unresolved
    (strings, None) -- they are resolved here, so the jax re-reduce path can
    replay a PDForward's stored fields directly.
    """
    if dualize is None:
        dualize = default_dualize_for_filtration(under_fil)

    if dims_to_backprop is None:
        # Cover all simplex dims so partial-U is admissible everywhere.
        # For H_k pairs the birth simplex has dim k and the death simplex
        # has dim k+1, so we need range(max_dim + 1).
        dims_to_backprop = list(range(max_dim + 1))

    n_threads = max(1, int(n_threads) if n_threads is not None else 1)
    strategy = resolve_strategy(conflict_strategy)
    u_strategy = resolve_u_strategy(u_strategy)

    top_opt = TopologyOptimizer(
        under_fil,
        with_crit_sets=(method == "crit-sets"),
        dims_to_restore_elz=dims_to_backprop,
        n_threads=n_threads,
        u_strategy=u_strategy,
    )
    decmp = _forward_reduce(top_opt, dualize)
    nondiff_dgms = decmp.diagram(under_fil, include_inf_points=False)
    index_dgm = {
        dim: nondiff_dgms.index_diagram_in_dimension(
            dim, as_numpy=True).astype(np.int64)
        for dim in range(max_dim)
    }
    return PDForward(
        top_opt=top_opt,
        values_np=np.asarray(values_np),
        index_dgm=index_dgm,
        negate=bool(under_fil.negate),
        dualize=dualize,
        method=method,
        step_size=step_size,
        strategy=strategy,
        max_dim=max_dim,
        dims_to_backprop=dims_to_backprop,
        n_threads=n_threads,
        u_strategy=u_strategy,
    )


def _backward_dgm_loss(fwd, dim, grad_np):
    grad_vals = np.zeros_like(fwd.values_np)
    index_dgm = fwd.index_dgm[dim]
    if index_dgm.size == 0:
        return grad_vals
    np.add.at(grad_vals, index_dgm[:, 0], grad_np[:, 0])
    np.add.at(grad_vals, index_dgm[:, 1], grad_np[:, 1])
    return grad_vals


def _backward_crit_sets(fwd, dim, grad_np):
    index_dgm = fwd.index_dgm[dim]
    fil_values = fwd.values_np
    top_opt = fwd.top_opt
    negate = fwd.negate

    grad_vals = np.zeros_like(fil_values)
    if index_dgm.size == 0:
        return grad_vals

    b_idx = index_dgm[:, 0]
    d_idx = index_dgm[:, 1]
    b_cur = fil_values[b_idx]
    d_cur = fil_values[d_idx]
    b_tgt = b_cur - fwd.step_size * grad_np[:, 0]
    d_tgt = d_cur - fwd.step_size * grad_np[:, 1]
    b_move = b_tgt != b_cur
    d_move = d_tgt != d_cur

    v_hom, u_hom, v_coh, u_coh = determine_needed_matrices(grad_np, negate)

    if v_hom or u_hom:
        top_opt.ensure_hom_reduced()
    if v_coh or u_coh:
        top_opt.ensure_coh_reduced()

    if u_hom:
        rows, bounds = select_u_moves(d_idx, d_cur, d_tgt, d_move,
                                      side="hom", negate=negate)
        top_opt.ensure_has_u_hom(dim, rows, bounds)
    if u_coh:
        rows, bounds = select_u_moves(b_idx, b_cur, b_tgt, b_move,
                                      side="coh", negate=negate)
        top_opt.ensure_has_u_coh(dim, rows, bounds)

    flat_idx = np.concatenate([b_idx[b_move], d_idx[d_move]])
    flat_tgt = np.concatenate([b_tgt[b_move], d_tgt[d_move]])
    if flat_idx.size == 0:
        return grad_vals

    # crit_sets_apply handles the dispatch reduction (ensure_hom_reduced)
    # internally and raises if the optimizer is dgm-loss only.
    indvals = top_opt.crit_sets_apply(flat_idx.astype(np.uintp).tolist(),
                                      flat_tgt.tolist(), fwd.strategy)
    out_idx = np.asarray(indvals.indices_array(), copy=True).astype(np.int64)
    # cast the C++ targets to the values dtype BEFORE subtracting, as the
    # torch path always did: keeps float32 values on a float64 backend
    # bit-identical to the pre-refactor behavior
    out_tgt = np.asarray(indvals.values_array()).astype(fil_values.dtype)
    if out_idx.size == 0:
        return grad_vals

    if fwd.strategy == _oineus.ConflictStrategy.Sum:
        np.add.at(grad_vals, out_idx, fil_values[out_idx] - out_tgt)
    else:
        grad_vals[out_idx] = fil_values[out_idx] - out_tgt
    return grad_vals


def pd_backward(fwd, dim, grad_np):
    """Gradient on the filtration values (numpy, shaped like values_np)
    from the upstream diagram gradient grad_np (numpy ``(n_d, 2)``)."""
    if fwd.method == "dgm-loss":
        return _backward_dgm_loss(fwd, dim, grad_np)
    if fwd.method == "crit-sets":
        return _backward_crit_sets(fwd, dim, grad_np)
    raise RuntimeError(f"Unknown gradient method: {fwd.method}")


# ---------------------------------------------------------------------------
# numpy port of the conflict-resolving aggregation (see _combine.py for the
# torch counterpart and the benchmark notes on when each one wins)
# ---------------------------------------------------------------------------

def critical_sets_to_flat_np(critical_sets):
    """Flatten C++-shaped CriticalSets [(target, [ids]), ...] to two numpy
    arrays (int64 ids with repeats, float64 per-id targets)."""
    if len(critical_sets) == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))

    sizes = np.fromiter((len(cs[1]) for cs in critical_sets),
                        dtype=np.int64, count=len(critical_sets))
    total = int(sizes.sum())
    if total == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))

    target_per_set = np.fromiter((cs[0] for cs in critical_sets),
                                 dtype=np.float64, count=len(critical_sets))
    flat_idx = np.empty(total, dtype=np.int64)
    pos = 0
    for cs, n in zip(critical_sets, sizes):
        if n:
            flat_idx[pos:pos + n] = cs[1]
            pos += n
    flat_tgt = np.repeat(target_per_set, sizes)
    return flat_idx, flat_tgt


def combine(flat_indices, flat_targets, strategy, current_values=None,
            target_map=None):
    """Resolve conflicts and return (indices, targets) numpy arrays.

    numpy port of the C++ combine_loss / the torch _combine.combine:

    flat_indices: (M,) int simplex (sorted) ids, possibly with repeats.
    flat_targets: (M,) real target value contributed for that id.
    strategy: _oineus.ConflictStrategy or one of "avg"/"max"/"sum"/"fca".
    current_values: (N,) real, indexed by simplex id; required for "max".
    target_map: dict[int, float], required for "fca". Maps a critical
        simplex's sorted id to its prescribed target value.

    For "sum" the returned arrays preserve duplicates -- the caller is
    expected to scatter-add into the per-simplex gradient.
    """
    strategy = resolve_strategy(strategy)
    flat_indices = np.asarray(flat_indices)
    flat_targets = np.asarray(flat_targets)
    if flat_indices.size == 0:
        return flat_indices, flat_targets

    if strategy == _oineus.ConflictStrategy.Sum:
        return flat_indices, flat_targets

    unique_ids, inverse = np.unique(flat_indices, return_inverse=True)
    n_groups = unique_ids.shape[0]
    dtype = flat_targets.dtype

    if strategy == _oineus.ConflictStrategy.Avg:
        sums = np.zeros(n_groups, dtype=dtype)
        np.add.at(sums, inverse, flat_targets)
        counts = np.zeros(n_groups, dtype=dtype)
        np.add.at(counts, inverse, 1)
        return unique_ids, sums / counts

    if strategy == _oineus.ConflictStrategy.Max:
        if current_values is None:
            raise ValueError("Max requires current_values")
        disp = np.abs(flat_targets - np.asarray(current_values)[flat_indices])
        max_disp = np.full(n_groups, -1.0, dtype=dtype)
        np.maximum.at(max_disp, inverse, disp)
        is_max = disp == max_disp[inverse]
        # Resolve ties deterministically: pick the smallest position in the
        # flat batch among entries that achieve the per-group max.
        sentinel = flat_indices.shape[0] + 1
        positions = np.arange(flat_indices.shape[0])
        cand = np.where(is_max, positions, sentinel)
        picked = np.full(n_groups, sentinel, dtype=positions.dtype)
        np.minimum.at(picked, inverse, cand)
        return unique_ids, flat_targets[picked]

    if strategy == _oineus.ConflictStrategy.FixCritAvg:
        if target_map is None:
            raise ValueError("FixCritAvg requires target_map")
        sums = np.zeros(n_groups, dtype=dtype)
        np.add.at(sums, inverse, flat_targets)
        counts = np.zeros(n_groups, dtype=dtype)
        np.add.at(counts, inverse, 1)
        avg = sums / counts
        if target_map:
            override_keys = np.fromiter(target_map.keys(), dtype=np.int64,
                                        count=len(target_map))
            override_vals = np.fromiter(target_map.values(), dtype=dtype,
                                        count=len(target_map))
            pos = np.searchsorted(unique_ids, override_keys)
            in_range = pos < unique_ids.shape[0]
            pos_clamped = np.minimum(pos, unique_ids.shape[0] - 1)
            hit = in_range & (unique_ids[pos_clamped] == override_keys)
            if hit.any():
                avg[pos[hit]] = override_vals[hit]
        return unique_ids, avg

    raise ValueError(f"unsupported strategy {strategy!r}")


# ---------------------------------------------------------------------------
# kernel/image/cokernel (KICR): framework-neutral forward
# ---------------------------------------------------------------------------

# C++ sentinel for "no death cell" in an index diagram entry: points at
# infinity carry k_invalid_index / plus_inf (both size_t max, common_defs.h)
# as their death index
KICR_INVALID_INDEX = np.iinfo(np.uint64).max

KICR_FAMILIES = ("kernel", "image", "cokernel")


@dataclass
class KICRForward:
    """Result of the KICR forward reduction: everything the gathers need.

    All indices are sorted ids of the FULL filtration K. kernel.h reads
    every birth/death value of every family through
    fil_K_.value_by_sorted_id -- including kernel death cells, which live
    in L but enter the diagram as sorted_L_to_sorted_K_[tau] -- so each
    per-dimension index diagram subscripts K's values tensor directly.
    Points at infinity (death index == KICR_INVALID_INDEX) are filtered
    out here.
    """
    kicr: object       # live C++ KerImCokReduced
    index_dgms: dict   # family -> {dim -> (n_d, 2) int64 [birth_sid_K, death_sid_K]}
    families: tuple    # the subset of KICR_FAMILIES that was computed
    max_dim: int


def kicr_forward(fil_K, fil_L, *, kernel, image, cokernel,
                 include_zero_persistence, n_threads):
    """Run the C++ ker/im/cok reduction for the inclusion L -> K and
    collect the finite index diagrams of the requested families as numpy.

    fil_K, fil_L are C++ filtrations; L must be a subcomplex of K carrying
    the same values on shared cells (the CEHM setting g = f restricted to
    L). The returned index diagrams are pure gather maps: the diagram in
    each dimension is values_K[index_dgm], so gradients reach K's values
    through the native VJP of the gather (a scatter-add) and no custom
    backward is needed. Zero-persistence pairs are dropped unless
    include_zero_persistence is True, mirroring the non-diff KICR
    diagrams.
    """
    # late import: the facade lives in the package __init__, which is fully
    # initialized by the time oineus.diff loads, but keeping the import here
    # makes pd_core importable in isolation for tests
    from .. import compute_kernel_image_cokernel_reduction

    families = tuple(name for name, on in
                     (("kernel", kernel), ("image", image), ("cokernel", cokernel))
                     if on)
    if not families:
        raise ValueError(
            "at least one of kernel/image/cokernel must be requested")

    params = _oineus.KICRParams()
    params.codomain = False
    params.kernel = bool(kernel)
    params.image = bool(image)
    params.cokernel = bool(cokernel)
    params.include_zero_persistence = bool(include_zero_persistence)
    params.n_threads = max(1, int(n_threads) if n_threads is not None else 1)

    kicr = compute_kernel_image_cokernel_reduction(fil_K, fil_L, params)

    # finite points of every family live in dims [0, max_dim): a finite pair
    # always involves a (dim+1)-cell (the death cell for image/cokernel, both
    # cells for kernel), so range(max_dim) covers them all -- same convention
    # as the ordinary differentiable diagrams
    max_dim = int(fil_K.max_dim)
    getter = {"kernel": kicr.kernel_diagrams, "image": kicr.image_diagrams,
              "cokernel": kicr.cokernel_diagrams}
    index_dgms = {}
    for family in families:
        dgms = getter[family]()
        by_dim = {}
        for dim in range(max_dim):
            arr = np.asarray(dgms.index_diagram_in_dimension(dim, as_numpy=True))
            arr = arr.reshape(-1, 2)
            finite = arr[:, 1] != KICR_INVALID_INDEX
            by_dim[dim] = arr[finite].astype(np.int64)
        index_dgms[family] = by_dim

    return KICRForward(kicr=kicr, index_dgms=index_dgms,
                       families=families, max_dim=max_dim)
