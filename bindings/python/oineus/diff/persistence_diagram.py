from .. import _oineus
import torch
import numpy as np

from .diff_filtration import DiffFiltration


_STRATEGY_MAP = {
    "avg": _oineus.ConflictStrategy.Avg,
    "max": _oineus.ConflictStrategy.Max,
    "sum": _oineus.ConflictStrategy.Sum,
    "fca": _oineus.ConflictStrategy.FixCritAvg,
}


def _resolve_strategy(strategy):
    if isinstance(strategy, _oineus.ConflictStrategy):
        return strategy
    try:
        return _STRATEGY_MAP[strategy.lower()]
    except (AttributeError, KeyError):
        raise ValueError(
            f"unknown conflict_strategy {strategy!r}; expected one of "
            f"{sorted(_STRATEGY_MAP)} or an _oineus.ConflictStrategy"
        )


# Phase-3 helpers: derive (cols, bounds) for compute_partial_u_from_v_1
# from the per-direction pair subsets that need U.
#
# Hom side (increase_death):
#   Walker reads u_data_t[d_idx_matrix] (= row d_idx of U_hom, since
#   hom is non-dualize so matrix_idx == filtration_idx). For each
#   (d_p, target_death) with target_death > current_death (non-negate)
#   or target_death < current_death (negate), the walker visits tau_matrix
#   in [d_p_matrix, last-tau-in-dim-with-value-le-target_death].
#   cmp = "below" for non-negate (pivots have decreasing value, stop
#   when piv < value(d_p)); deepest bound = min(value(d_p)) per col.
#
# Coh side (decrease_birth):
#   Walker reads u_data_t[b_idx_matrix] where b_idx_matrix =
#   fil_size - b_p_filtration - 1. Cols range on coh: walk forward
#   in matrix order (= backward in filtration order) from b_p_matrix
#   until filtration value drops below target_birth. cmp = "above"
#   for non-negate (pivot filtration values increase along iteration);
#   deepest bound = max(value(b_p)) per col.

def _derive_cols_bounds_increase_death(fil, decmp_hom, d_idx_t, d_cur_t,
                                       d_tgt_t, d_move_t, negate):
    """Hom side: cols/bounds for the increase_death subset of moves.

    Returns (cols, bounds) ready for compute_partial_u_from_v_1.
    Skips pairs in the *opposite* direction (decrease_death uses V, no U).

    Bounds use C++ filtration values (fil.simplex_value_by_sorted_id),
    not the differentiable torch values, so that cmp_op compares
    against the same numeric scale that the partial driver's value_at
    callback returns. The torch fil.values can differ from the C++
    underlying values by ~1e-7 (max-distance recomputation drift),
    enough to make the first iteration of compute_u_column_1_bounded
    spuriously truncate when bound == value(d_p)."""
    if negate:
        # filtration-increasing on hom = death value decreasing in non-negate
        # sense; the partial-U path supports non-negate only for now.
        return [], []
    cols_to_bound = {}
    n_pairs = d_idx_t.shape[0]
    d_idx_np = d_idx_t.detach().cpu().numpy()
    d_cur_np = d_cur_t.detach().cpu().numpy()
    d_tgt_np = d_tgt_t.detach().cpu().numpy()
    d_move_np = d_move_t.detach().cpu().numpy()
    for i in range(n_pairs):
        if not d_move_np[i]:
            continue
        # Use torch values only for direction/target comparisons;
        # the partial-U bound needs the C++ filtration value.
        if d_tgt_np[i] <= d_cur_np[i]:
            continue
        d_p = int(d_idx_np[i])
        dv = fil.simplex_value_by_sorted_id(d_p)  # C++ value
        dt = float(d_tgt_np[i])
        dim_idx = _find_dim(decmp_hom, d_p)
        if dim_idx is None:
            continue
        dim_last = decmp_hom.dim_last[dim_idx]
        for c in range(d_p, dim_last + 1):
            v_c = fil.simplex_value_by_sorted_id(c)
            if v_c > dt:
                break
            cur = cols_to_bound.get(c)
            if cur is None or cur > dv:
                cols_to_bound[c] = dv
    cols = sorted(cols_to_bound.keys())
    bounds = [cols_to_bound[c] for c in cols]
    return cols, bounds


def _derive_cols_bounds_decrease_birth(fil, decmp_coh, b_idx_t, b_cur_t,
                                       b_tgt_t, b_move_t, negate):
    """Coh side: cols/bounds for the decrease_birth subset of moves.
    Same float-drift caveat as the hom helper: bounds come from C++
    filtration values, not torch fil.values."""
    if negate:
        return [], []
    fil_size = fil.size()
    cols_to_bound = {}
    n_pairs = b_idx_t.shape[0]
    b_idx_np = b_idx_t.detach().cpu().numpy()
    b_cur_np = b_cur_t.detach().cpu().numpy()
    b_tgt_np = b_tgt_t.detach().cpu().numpy()
    b_move_np = b_move_t.detach().cpu().numpy()
    for i in range(n_pairs):
        if not b_move_np[i]:
            continue
        if b_tgt_np[i] >= b_cur_np[i]:
            continue
        b_p_fil = int(b_idx_np[i])
        bv = fil.simplex_value_by_sorted_id(b_p_fil)  # C++ value
        bt = float(b_tgt_np[i])
        b_p_matrix = fil_size - b_p_fil - 1
        dim_idx = _find_dim(decmp_coh, b_p_matrix)
        if dim_idx is None:
            continue
        dim_last = decmp_coh.dim_last[dim_idx]
        for c in range(b_p_matrix, dim_last + 1):
            c_fil = fil_size - c - 1
            v_c = fil.simplex_value_by_sorted_id(c_fil)
            if v_c < bt:
                break
            cur = cols_to_bound.get(c)
            if cur is None or cur < bv:
                cols_to_bound[c] = bv
    cols = sorted(cols_to_bound.keys())
    bounds = [cols_to_bound[c] for c in cols]
    return cols, bounds


def _find_dim(decmp, matrix_idx):
    """Return the dim index whose [dim_first, dim_last] range contains
    matrix_idx, or None."""
    df = decmp.dim_first
    dl = decmp.dim_last
    for d in range(len(df)):
        if df[d] <= matrix_idx <= dl[d]:
            return d
    return None


# Phase-4 helpers: derive (rows, bounds, dim) for compute_partial_u_rows.
# Each pair contributes exactly one row index (the death-creator on hom,
# the birth-creator's matrix index on coh) and one bound (target value).
# Bounds are passed through as-is from torch -- the Phase-3 float-drift
# trap is gentler here (truncation off-by-one is tolerable) but we still
# bias toward C++ values for safety: see derivation below.
PHASE4_PARTIAL_THRESHOLD = 0.75


def _classify_increase_death_rows(fil, decmp_hom, d_idx_t, d_cur_t,
                                  d_tgt_t, d_move_t, negate):
    """For each death-up move (target > current on non-negate), emit
    (row_idx = d_p_filtration, bound = max(value(d_p), target_death)).
    The bound is in C++ filtration units; torch d_tgt may drift by ~1e-7
    from the C++ value of any cell, so we sandwich the bound between
    value(d_p) (always >= it suffices) and the torch target. For Phase-4
    increase_death walker (cmp=above), bound is passed as-is; the
    walker stops at piv_value > bound, which can be any value strictly
    greater than value(d_p) because the diagonal r is always emitted
    by the row primitive's first iteration."""
    if negate:
        return [], [], None
    rows, bounds = [], []
    common_dim = None
    d_idx_np = d_idx_t.detach().cpu().numpy()
    d_cur_np = d_cur_t.detach().cpu().numpy()
    d_tgt_np = d_tgt_t.detach().cpu().numpy()
    d_move_np = d_move_t.detach().cpu().numpy()
    for i in range(d_idx_t.shape[0]):
        if not d_move_np[i]:
            continue
        if d_tgt_np[i] <= d_cur_np[i]:
            continue
        d_p = int(d_idx_np[i])
        dim = _find_dim(decmp_hom, d_p)
        if dim is None:
            continue
        if common_dim is None:
            common_dim = dim
        elif dim != common_dim:
            continue
        rows.append(d_p)
        bounds.append(float(d_tgt_np[i]))
    return rows, bounds, common_dim


def _classify_decrease_birth_rows(fil, decmp_coh, b_idx_t, b_cur_t,
                                  b_tgt_t, b_move_t, negate):
    """Same shape but on coh side. Row index is the birth-creator's
    matrix index in the COH decomposition (= fil_size - b_p_fil - 1).
    Walker iterates in coh-matrix order which corresponds to
    decreasing filtration value, so cmp=below is the truncation
    direction; bound is the target_birth."""
    if negate:
        return [], [], None
    fil_size = fil.size()
    rows, bounds = [], []
    common_dim = None
    b_idx_np = b_idx_t.detach().cpu().numpy()
    b_cur_np = b_cur_t.detach().cpu().numpy()
    b_tgt_np = b_tgt_t.detach().cpu().numpy()
    b_move_np = b_move_t.detach().cpu().numpy()
    for i in range(b_idx_t.shape[0]):
        if not b_move_np[i]:
            continue
        if b_tgt_np[i] >= b_cur_np[i]:
            continue
        b_p_fil = int(b_idx_np[i])
        b_p_matrix = fil_size - b_p_fil - 1
        dim = _find_dim(decmp_coh, b_p_matrix)
        if dim is None:
            continue
        if common_dim is None:
            common_dim = dim
        elif dim != common_dim:
            continue
        rows.append(b_p_matrix)
        bounds.append(float(b_tgt_np[i]))
    return rows, bounds, common_dim


class PersistenceDiagramHelper(torch.autograd.Function):
    """
    Autograd function for extracting a single dimension's diagram.

    Forward: subscripts fil.values at birth/death indices.
    Backward:
        - "dgm-loss": gradients only on the two simplices that define each pair.
        - "crit-sets": gradients on the full critical set of every moved pair,
          conflicts resolved via Max/Avg/Sum/FixCritAvg.
    """

    @staticmethod
    def forward(ctx, fil_values, fil, top_opt, dgms, dim, include_inf_points,
                gradient_method, step_size, conflict_strategy):
        """
        Extract diagram for dimension `dim` as a differentiable tensor.

        `top_opt` carries the reduction state. For "dgm-loss" it is a
        single Decomposition (homology or cohomology). For "crit-sets" it
        is a TopologyOptimizer with both decompositions and U/V matrices
        already materialised.
        """
        index_dgm = dgms.index_diagram_in_dimension(dim, as_numpy=True).astype(np.int64)
        index_dgm = torch.from_numpy(index_dgm).to(fil_values.device)

        fil_len = fil_values.shape[0]

        ctx.gradient_method = gradient_method
        ctx.fil_len = fil_len
        ctx.include_inf_points = include_inf_points
        ctx.step_size = step_size
        ctx.dim = dim
        ctx.top_opt = top_opt
        ctx.fil = fil
        ctx.conflict_strategy = conflict_strategy
        ctx.negate = bool(fil.negate)
        ctx.n_threads = getattr(fil, "_phase3_n_threads", 1)

        n_total = len(index_dgm)

        if n_total == 0:
            ctx.save_for_backward(index_dgm, fil_values)
            return torch.zeros((0, 2), dtype=fil_values.dtype, device=fil_values.device)

        if include_inf_points:
            fin_mask = (index_dgm[:, 0] >= 0) & (index_dgm[:, 0] < fil_len) & (index_dgm[:, 1] >= 0) & (index_dgm[:, 1] < fil_len)
            fin_idx_dgm = index_dgm[fin_mask]
            inf_births_inds = index_dgm[~fin_mask][:, 0]
            finite_dgm = fil_values[fin_idx_dgm]
            inf_births = fil_values[inf_births_inds]
            inf_deaths = torch.full_like(inf_births, float('inf'), dtype=fil_values.dtype, device=fil_values.device)
            inf_dgm = torch.stack([inf_births, inf_deaths], dim=1)
            diagram = torch.cat([finite_dgm, inf_dgm], dim=0)
        else:
            diagram = fil_values[index_dgm]

        ctx.save_for_backward(index_dgm, fil_values)

        return diagram

    @staticmethod
    def backward(ctx, grad_output):
        index_dgm, fil_values = ctx.saved_tensors
        fil_len = ctx.fil_len

        if ctx.gradient_method == "dgm-loss":
            grad_vals = torch.zeros(fil_len, dtype=grad_output.dtype, device=grad_output.device)
            grad_vals[index_dgm.flatten()] = grad_output.flatten()
            return grad_vals, None, None, None, None, None, None, None, None

        if ctx.gradient_method not in ("crit-sets", "crit-sets-partial",
                                       "crit-sets-row-partial"):
            raise RuntimeError(f"unknown gradient_method {ctx.gradient_method!r}")

        # Crit-sets path. Restrict to finite pairs; rows of grad_output match
        # the layout produced by forward (finite pairs first when
        # include_inf_points is True). For include_inf_points=False every
        # row is finite by construction.
        n_total = index_dgm.shape[0]
        device = grad_output.device
        grad_vals = torch.zeros(fil_len, dtype=grad_output.dtype, device=device)

        if n_total == 0:
            return grad_vals, None, None, None, None, None, None, None, None

        if ctx.include_inf_points:
            fin_mask = (index_dgm[:, 0] >= 0) & (index_dgm[:, 0] < fil_len) & (index_dgm[:, 1] >= 0) & (index_dgm[:, 1] < fil_len)
            n_fin = int(fin_mask.sum().item())
            fin_idx_dgm = index_dgm[fin_mask]
            fin_grad = grad_output[:n_fin]
        else:
            fin_idx_dgm = index_dgm
            fin_grad = grad_output

        if fin_idx_dgm.shape[0] == 0:
            return grad_vals, None, None, None, None, None, None, None, None

        b_idx = fin_idx_dgm[:, 0]
        d_idx = fin_idx_dgm[:, 1]
        b_cur = fil_values[b_idx]
        d_cur = fil_values[d_idx]
        step = ctx.step_size
        b_tgt = b_cur - step * fin_grad[:, 0]
        d_tgt = d_cur - step * fin_grad[:, 1]
        b_move = b_tgt != b_cur
        d_move = d_tgt != d_cur

        # Classify the requested moves to decide which decompositions and
        # which U matrices we actually need. The C++ side compares
        # current vs target via fil.cmp() which respects negate; we
        # mirror that here so we ask for U on a side only when at least
        # one move on that side reads U.
        negate = bool(ctx.negate)
        if negate:
            # filtration-increasing means value-decreasing
            need_u_hom = bool((d_tgt[d_move] < d_cur[d_move]).any().item()) if d_move.any() else False
            need_u_coh = bool((b_tgt[b_move] > b_cur[b_move]).any().item()) if b_move.any() else False
        else:
            need_u_hom = bool((d_tgt[d_move] > d_cur[d_move]).any().item()) if d_move.any() else False
            need_u_coh = bool((b_tgt[b_move] < b_cur[b_move]).any().item()) if b_move.any() else False

        any_death = bool(d_move.any().item())
        any_birth = bool(b_move.any().item())

        top_opt = ctx.top_opt
        n_threads = max(1, int(getattr(ctx, "n_threads", 1) or 1))
        use_partial = (ctx.gradient_method == "crit-sets-partial")
        use_phase4 = (ctx.gradient_method == "crit-sets-row-partial")

        if use_phase4:
            # Phase-4: V-only reductions; partial-U via row solves on V^T.
            under_fil = ctx.fil.under_fil if hasattr(ctx.fil, "under_fil") else ctx.fil
            if any_death:
                if need_u_hom:
                    top_opt.ensure_reduced_for_partial_u_hom(n_threads)
                    decmp_hom = top_opt.homology_decomposition_ref()
                    rows_h, bounds_h, dim_h = _classify_increase_death_rows(
                        ctx.fil, decmp_hom, d_idx, d_cur, d_tgt, d_move,
                        negate=negate)
                    if rows_h:
                        dim_size = (decmp_hom.dim_last[dim_h]
                                    - decmp_hom.dim_first[dim_h] + 1)
                        if len(rows_h) / dim_size > PHASE4_PARTIAL_THRESHOLD:
                            decmp_hom.compute_full_u_rows(
                                under_fil, dim_h, n_threads=n_threads)
                        else:
                            decmp_hom.compute_partial_u_rows(
                                under_fil, rows_h, bounds_h, dim_h,
                                cmp=("below" if negate else "above"),
                                n_threads=n_threads)
                else:
                    top_opt.ensure_reduced_hom(need_u=False)
            if any_birth:
                if need_u_coh:
                    top_opt.ensure_reduced_for_partial_u_coh(n_threads)
                    decmp_coh = top_opt.cohomology_decomposition_ref()
                    rows_c, bounds_c, dim_c = _classify_decrease_birth_rows(
                        ctx.fil, decmp_coh, b_idx, b_cur, b_tgt, b_move,
                        negate=negate)
                    if rows_c:
                        dim_size = (decmp_coh.dim_last[dim_c]
                                    - decmp_coh.dim_first[dim_c] + 1)
                        if len(rows_c) / dim_size > PHASE4_PARTIAL_THRESHOLD:
                            decmp_coh.compute_full_u_rows(
                                under_fil, dim_c, n_threads=n_threads)
                        else:
                            decmp_coh.compute_partial_u_rows(
                                under_fil, rows_c, bounds_c, dim_c,
                                cmp=("above" if negate else "below"),
                                n_threads=n_threads)
                else:
                    top_opt.ensure_reduced_coh(need_u=False)
        elif use_partial:
            # Phase-3: V-only reductions; partial-U via cols/bounds
            # derived from the U-needing pair subset on each side.
            under_fil = ctx.fil.under_fil if hasattr(ctx.fil, "under_fil") else ctx.fil
            if any_death:
                if need_u_hom:
                    top_opt.ensure_reduced_for_partial_u_hom(n_threads)
                    decmp_hom = top_opt.homology_decomposition_ref()
                    cols_h, bounds_h = _derive_cols_bounds_increase_death(
                        ctx.fil, decmp_hom, d_idx, d_cur, d_tgt, d_move,
                        negate=negate)
                    if cols_h:
                        decmp_hom.compute_partial_u_from_v_1(
                            under_fil, cols_h, bounds_h,
                            cmp=("above" if negate else "below"),
                            n_threads=n_threads)
                else:
                    top_opt.ensure_reduced_hom(need_u=False)
            if any_birth:
                if need_u_coh:
                    top_opt.ensure_reduced_for_partial_u_coh(n_threads)
                    decmp_coh = top_opt.cohomology_decomposition_ref()
                    cols_c, bounds_c = _derive_cols_bounds_decrease_birth(
                        ctx.fil, decmp_coh, b_idx, b_cur, b_tgt, b_move,
                        negate=negate)
                    if cols_c:
                        decmp_coh.compute_partial_u_from_v_1(
                            under_fil, cols_c, bounds_c,
                            cmp=("below" if negate else "above"),
                            n_threads=n_threads)
                else:
                    top_opt.ensure_reduced_coh(need_u=False)
        else:
            # Phase-2 path: ensure_reduced_* with the right need_u flag
            # and a clearing-off / in-band U reduction when needed.
            if any_death:
                top_opt.ensure_reduced_hom(need_u=need_u_hom)
            if any_birth:
                top_opt.ensure_reduced_coh(need_u=need_u_coh)

        flat_idx = torch.cat([b_idx[b_move], d_idx[d_move]])
        flat_tgt = torch.cat([b_tgt[b_move], d_tgt[d_move]])

        if flat_idx.numel() == 0:
            return grad_vals, None, None, None, None, None, None, None, None

        flat_idx_np = flat_idx.detach().cpu().numpy().astype(np.uintp).tolist()
        flat_tgt_np = flat_tgt.detach().cpu().to(torch.float64).numpy().tolist()

        strategy = _resolve_strategy(ctx.conflict_strategy)
        indvals = top_opt.crit_sets_apply(flat_idx_np, flat_tgt_np, strategy)
        out_idx_np = np.asarray(indvals.indices_array(), copy=True)
        out_tgt_np = np.asarray(indvals.values_array(), copy=True)

        if out_idx_np.size == 0:
            return grad_vals, None, None, None, None, None, None, None, None

        idx_t = torch.from_numpy(out_idx_np.astype(np.int64)).to(device=device)
        tgt_t = torch.from_numpy(out_tgt_np).to(dtype=grad_output.dtype, device=device)

        if strategy == _oineus.ConflictStrategy.Sum:
            # Sum may emit duplicate indices: aggregate per-index gradients.
            contrib = fil_values[idx_t] - tgt_t
            grad_vals.scatter_add_(0, idx_t, contrib)
        else:
            grad_vals[idx_t] = fil_values[idx_t] - tgt_t

        return grad_vals, None, None, None, None, None, None, None, None


class PersistenceDiagrams:
    """
    Container for differentiable persistence diagrams in all dimensions.

    Usage:
        dgms = persistence_diagram(fil, dualize=True)
        dgm1 = dgms[1]  # H1 diagram as tensor (N, 2)
        loss = (dgm1[:, 1] - dgm1[:, 0]).pow(2).sum()
        loss.backward()
    """

    def __init__(self, fil: DiffFiltration, dualize, include_inf_points: bool,
                 gradient_method: str, step_size: float, conflict_strategy,
                 rp=None, n_threads=None):
        if not isinstance(fil.values, torch.Tensor):
            raise TypeError("fil.values must be a torch.Tensor for differentiable diagrams")

        if rp is None:
            rp = _oineus.ReductionParams()

        # If the caller did not pin dualize, prefer cohomology for VR
        # (where coh+clearing wins decisively); homology for everything
        # else.
        if dualize is None:
            dualize = (fil.under_fil.kind == _oineus.FiltrationKind.Vr)

        self._fil = fil
        self._dualize = dualize
        self._include_inf_points = include_inf_points
        self._gradient_method = gradient_method
        self._step_size = step_size
        self._conflict_strategy = conflict_strategy

        if gradient_method in ("crit-sets", "crit-sets-partial",
                               "crit-sets-row-partial"):
            # Phase-2 / Phase-3 share the same defer-reduction setup;
            # they differ only in what backward does (full U vs partial U).
            top_opt = _oineus.TopologyOptimizer(fil.under_fil, defer_reduction=True)
            nondiff_dgms = top_opt.compute_diagram(include_inf_points=include_inf_points)
            self._top_opt = top_opt
            if n_threads is not None:
                # Stash on the under_fil so backward (with no direct constructor
                # access) can read it via ctx.fil._phase3_n_threads.
                try:
                    fil.under_fil._phase3_n_threads = int(n_threads)
                except Exception:
                    pass
        elif gradient_method == "dgm-loss":
            kwargs = {"dualize": dualize}
            if n_threads is not None:
                kwargs["n_threads"] = n_threads
                # Also propagate into the reduction params; the default
                # ReductionParams() uses n_threads=1 which would hide
                # parallel-reduction wins on bigger inputs.
                rp.n_threads = int(n_threads)
            dcmp = _oineus.Decomposition(fil.under_fil, **kwargs)
            dcmp.reduce(rp)
            nondiff_dgms = dcmp.diagram(fil.under_fil, include_inf_points=include_inf_points)
            self._top_opt = dcmp
        else:
            raise ValueError(
                f"unknown gradient_method {gradient_method!r}; expected "
                "'dgm-loss', 'crit-sets', 'crit-sets-partial', or "
                "'crit-sets-row-partial'"
            )

        self._diagrams = {
            dim: PersistenceDiagramHelper.apply(
                self._fil.values,
                self._fil.under_fil,
                self._top_opt,
                nondiff_dgms,
                dim,
                self._include_inf_points,
                self._gradient_method,
                self._step_size,
                self._conflict_strategy,
            )
            for dim in range(self._fil.max_dim())
        }

    def __getitem__(self, dim: int) -> torch.Tensor:
        if dim not in self._diagrams:
            raise KeyError(f"No diagram for dimension {dim}. Available: {list(self._diagrams.keys())}")
        return self._diagrams[dim]

    def __contains__(self, dim: int) -> bool:
        return dim in self._diagrams

    def __len__(self) -> int:
        return len(self._diagrams)

    def __iter__(self):
        return iter(self._diagrams)

    def keys(self):
        return self._diagrams.keys()

    def values(self):
        return self._diagrams.values()

    def items(self):
        return self._diagrams.items()

    def in_dimension(self, dim: int) -> torch.Tensor:
        return self[dim]

    @property
    def max_dim(self) -> int:
        return max(self._diagrams.keys())


def persistence_diagram(
    fil: DiffFiltration,
    dualize=None,
    include_inf_points: bool = False,
    gradient_method: str = "dgm-loss",
    step_size: float = 1.0,
    conflict_strategy="avg",
    n_threads=None,
) -> PersistenceDiagrams:
    """
    Compute differentiable persistence diagrams from a DiffFiltration.

    Args:
        fil: DiffFiltration with differentiable `values` tensor.
        dualize: cohomology if True, homology if False. None (default)
            picks cohomology for VR filtrations (where coh+clearing
            wins decisively) and homology otherwise. Used only for
            "dgm-loss"; "crit-sets" reduces each side lazily based on
            the moves it actually sees.
        include_inf_points: if True, include infinite points (deaths
            set to float('inf')); only finite pairs receive crit-sets
            gradient.
        gradient_method: "dgm-loss" or "crit-sets".
        step_size: scales grad_output to a target diagram (target =
            current - step_size * grad_output) for the "crit-sets"
            pass. Ignored for "dgm-loss".
        conflict_strategy: "avg", "max", "sum", or "fca", or any
            _oineus.ConflictStrategy. Used only for "crit-sets".
        n_threads: passed to the underlying decomposition for "dgm-loss".

    Returns:
        PersistenceDiagrams: dict-like, dim -> Tensor (N, 2). Gradients
        flow back to fil.values.
    """
    return PersistenceDiagrams(
        fil=fil,
        dualize=dualize,
        include_inf_points=include_inf_points,
        gradient_method=gradient_method,
        step_size=step_size,
        conflict_strategy=conflict_strategy,
        n_threads=n_threads,
    )
