from .. import _oineus
import torch
import numpy as np

from .diff_filtration import DiffFiltration
from .top_optimizer import make_under_topology_optimizer


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


# U-computation strategies for the crit-sets backward.
#
#   auto           -- production default. Currently resolves to
#                     row_partial with a partial-vs-full threshold
#                     dispatch (see ROW_PARTIAL_FULL_FALLBACK_THRESHOLD).
#                     Use this in production code.
#   row_partial    -- Row-form partial U: solve (row r of U) V = e_r^T
#                     directly for each row the walker reads (one
#                     parallel transpose of V to get V^T, then
#                     embarrassingly parallel row solves with value-
#                     bound truncation). Falls back to compute_full_u_rows
#                     when n_pairs / dim_size > the threshold above.
#   legacy_in_band -- in-band U built during reduction (clearing off,
#                     compute_u=true; serial reduction). Available as
#                     a control / cross-check strategy. The original
#                     in-band ELZ algorithm.

_VALID_U_STRATEGIES = ("auto", "legacy_in_band", "row_partial")


def _resolve_u_strategy(u_strategy, gradient_method):
    """Pick the effective u_strategy. Explicit parameter wins over
    gradient_method. Default is 'auto'."""
    if u_strategy is None:
        return "auto"
    if u_strategy not in _VALID_U_STRATEGIES:
        raise ValueError(
            f"unknown u_strategy {u_strategy!r}; expected one of "
            f"{_VALID_U_STRATEGIES}")
    return u_strategy


def _dispatch_u_for_side(top_opt, under_fil, side, *, need_u, any_move,
                         u_strategy, move_idx_t, cur_t, tgt_t, move_mask_t,
                         negate, n_threads):
    """Dispatcher: ensure the requested U-computation has happened on
    `side` (= 'hom' or 'coh') so that crit_sets_apply's walker can read
    u_data_t. Each strategy follows the same outline:

      1. If no moves on this side -> nothing to do.
      2. If no moves need U on this side -> ensure V is reduced.
      3. Otherwise: ensure V (parallel + restore_elz, no U), then
         compute U via the strategy's choice.

    `move_idx_t` etc. are the per-side torch tensors of pair indices,
    current values, target values, and move masks (d_idx for hom,
    b_idx for coh).
    """
    if not any_move:
        return
    if not need_u:
        # No U needed on this side; just ensure V is reduced for the
        # walker (decrease_death / increase_birth read v_data only).
        if side == "hom":
            top_opt.ensure_reduced_hom(need_u=False)
        else:
            top_opt.ensure_reduced_coh(need_u=False)
        return

    if u_strategy == "legacy_in_band":
        # In-band U: clearing off, U built during reduction.
        if side == "hom":
            top_opt.ensure_reduced_hom(need_u=True)
        else:
            top_opt.ensure_reduced_coh(need_u=True)
        return

    if u_strategy not in ("auto", "row_partial"):
        raise RuntimeError(f"unhandled u_strategy {u_strategy!r}")

    # auto / row_partial: parallel V-only reduction with restore_elz,
    # followed by a row-form partial U-pass that auto-falls-back to
    # the full row-form pass when many rows are needed.
    if side == "hom":
        top_opt.ensure_reduced_for_partial_u_hom(n_threads)
        decmp = top_opt.homology_decomposition_ref()
        cmp_partial_row = "below" if negate else "above"
        rows, bounds, dim_u = _classify_increase_death_rows(
            under_fil, decmp, move_idx_t, cur_t, tgt_t, move_mask_t,
            negate=negate)
    else:
        top_opt.ensure_reduced_for_partial_u_coh(n_threads)
        decmp = top_opt.cohomology_decomposition_ref()
        cmp_partial_row = "above" if negate else "below"
        rows, bounds, dim_u = _classify_decrease_birth_rows(
            under_fil, decmp, move_idx_t, cur_t, tgt_t, move_mask_t,
            negate=negate)

    if dim_u is None:
        # No classifiable rows (e.g. negate filtration where the
        # row helpers bail). Fall back to a full row-form pass at
        # the dim of the first move.
        idx_np = move_idx_t.detach().cpu().numpy()
        mv_np = move_mask_t.detach().cpu().numpy()
        first_idx = None
        for i in range(idx_np.shape[0]):
            if mv_np[i]:
                first_idx = int(idx_np[i])
                break
        if first_idx is None:
            return
        if side == "coh":
            first_idx = under_fil.size() - first_idx - 1
        dim_u = _find_dim(decmp, first_idx)
        if dim_u is None:
            return
        decmp.compute_full_u_rows(under_fil, dim_u, n_threads=n_threads)
        return

    if not rows:
        decmp.compute_full_u_rows(under_fil, dim_u, n_threads=n_threads)
        return
    dim_size = decmp.dim_last[dim_u] - decmp.dim_first[dim_u] + 1
    if len(rows) / dim_size > ROW_PARTIAL_FULL_FALLBACK_THRESHOLD:
        decmp.compute_full_u_rows(under_fil, dim_u, n_threads=n_threads)
    else:
        decmp.compute_partial_u_rows(
            under_fil, rows, bounds, dim_u,
            cmp=cmp_partial_row, n_threads=n_threads)


def _find_dim(decmp, matrix_idx):
    """Return the dim index whose [dim_first, dim_last] range contains
    matrix_idx, or None."""
    df = decmp.dim_first
    dl = decmp.dim_last
    for d in range(len(df)):
        if df[d] <= matrix_idx <= dl[d]:
            return d
    return None


# Row-form partial-U helpers: derive (rows, bounds, dim) for
# compute_partial_u_rows. Each pair contributes exactly one row index
# (the death-creator on hom, the birth-creator's matrix index on coh)
# and one bound (the target value). Bounds may be passed through from
# torch directly: a tiny float drift between fil.values and the C++
# filtration values only over- or under-shoots the row solve by one
# iteration, which is harmless because the diagonal entry is always
# emitted by the row primitive's first iteration.
ROW_PARTIAL_FULL_FALLBACK_THRESHOLD = 0.75


def _classify_increase_death_rows(fil, decmp_hom, d_idx_t, d_cur_t,
                                  d_tgt_t, d_move_t, negate):
    """For each death-up move (target > current on non-negate), emit
    (row_idx = d_p_filtration, bound = target_death). The row solve's
    cmp_op is "above" (stop when piv_value > bound), so the iteration
    walks columns from d_p upward and stops once it crosses
    target_death. A small float drift between the torch target and
    the C++ filtration values at the boundary only over- or
    under-shoots the row solve by one iteration; the diagonal r is
    always emitted by the primitive's first iteration."""
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
                gradient_method, step_size, conflict_strategy, u_strategy=None):
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
        ctx.u_strategy = _resolve_u_strategy(u_strategy, gradient_method)
        ctx.fil_len = fil_len
        ctx.include_inf_points = include_inf_points
        ctx.step_size = step_size
        ctx.dim = dim
        ctx.top_opt = top_opt
        ctx.fil = fil
        ctx.conflict_strategy = conflict_strategy
        ctx.negate = bool(fil.negate)
        ctx.n_threads = getattr(fil, "_crit_sets_n_threads", 1)

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
            ctx.n_fin = int(fin_mask.sum().item())
            ctx.fin_idx_dgm = fin_idx_dgm
            ctx.inf_births_inds = inf_births_inds
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
            if ctx.include_inf_points:
                # Split forward layout: finite pairs first, then inf-death pairs.
                # Inf-death rows have an out-of-range index in the death column;
                # only birth indices are real. Death is the constant +inf, so
                # its gradient is dropped.
                n_fin = ctx.n_fin
                fin_idx_dgm = ctx.fin_idx_dgm
                inf_births_inds = ctx.inf_births_inds
                if fin_idx_dgm.shape[0] > 0:
                    grad_vals[fin_idx_dgm.flatten()] = grad_output[:n_fin].flatten()
                if inf_births_inds.shape[0] > 0:
                    grad_vals[inf_births_inds] = grad_output[n_fin:, 0]
            else:
                grad_vals[index_dgm.flatten()] = grad_output.flatten()
            return grad_vals, None, None, None, None, None, None, None, None, None

        if ctx.gradient_method != "crit-sets":
            # Be permissive: any gradient_method that isn't dgm-loss
            # falls through to the crit-sets backward, with the chosen
            # u_strategy controlling which U-computation is used.
            pass

        # Crit-sets path. Restrict to finite pairs; rows of grad_output match
        # the layout produced by forward (finite pairs first when
        # include_inf_points is True). For include_inf_points=False every
        # row is finite by construction.
        n_total = index_dgm.shape[0]
        device = grad_output.device
        grad_vals = torch.zeros(fil_len, dtype=grad_output.dtype, device=device)

        if n_total == 0:
            return grad_vals, None, None, None, None, None, None, None, None, None

        if ctx.include_inf_points:
            fin_mask = (index_dgm[:, 0] >= 0) & (index_dgm[:, 0] < fil_len) & (index_dgm[:, 1] >= 0) & (index_dgm[:, 1] < fil_len)
            n_fin = int(fin_mask.sum().item())
            fin_idx_dgm = index_dgm[fin_mask]
            fin_grad = grad_output[:n_fin]
        else:
            fin_idx_dgm = index_dgm
            fin_grad = grad_output

        if fin_idx_dgm.shape[0] == 0:
            return grad_vals, None, None, None, None, None, None, None, None, None

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
        u_strategy = getattr(ctx, "u_strategy", "auto")

        under_fil = ctx.fil.under_fil if hasattr(ctx.fil, "under_fil") else ctx.fil
        _dispatch_u_for_side(
            top_opt, under_fil, "hom",
            need_u=need_u_hom, any_move=any_death,
            u_strategy=u_strategy,
            move_idx_t=d_idx, cur_t=d_cur, tgt_t=d_tgt, move_mask_t=d_move,
            negate=negate, n_threads=n_threads)
        _dispatch_u_for_side(
            top_opt, under_fil, "coh",
            need_u=need_u_coh, any_move=any_birth,
            u_strategy=u_strategy,
            move_idx_t=b_idx, cur_t=b_cur, tgt_t=b_tgt, move_mask_t=b_move,
            negate=negate, n_threads=n_threads)

        flat_idx = torch.cat([b_idx[b_move], d_idx[d_move]])
        flat_tgt = torch.cat([b_tgt[b_move], d_tgt[d_move]])

        if flat_idx.numel() == 0:
            return grad_vals, None, None, None, None, None, None, None, None, None

        flat_idx_np = flat_idx.detach().cpu().numpy().astype(np.uintp).tolist()
        flat_tgt_np = flat_tgt.detach().cpu().to(torch.float64).numpy().tolist()

        strategy = _resolve_strategy(ctx.conflict_strategy)
        indvals = top_opt.crit_sets_apply(flat_idx_np, flat_tgt_np, strategy)
        out_idx_np = np.asarray(indvals.indices_array(), copy=True)
        out_tgt_np = np.asarray(indvals.values_array(), copy=True)

        if out_idx_np.size == 0:
            return grad_vals, None, None, None, None, None, None, None, None, None

        idx_t = torch.from_numpy(out_idx_np.astype(np.int64)).to(device=device)
        tgt_t = torch.from_numpy(out_tgt_np).to(dtype=grad_output.dtype, device=device)

        if strategy == _oineus.ConflictStrategy.Sum:
            # Sum may emit duplicate indices: aggregate per-index gradients.
            contrib = fil_values[idx_t] - tgt_t
            grad_vals.scatter_add_(0, idx_t, contrib)
        else:
            grad_vals[idx_t] = fil_values[idx_t] - tgt_t

        return grad_vals, None, None, None, None, None, None, None, None, None


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
                 rp=None, n_threads=None, u_strategy=None):
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

        if gradient_method == "crit-sets":
            top_opt = make_under_topology_optimizer(fil.under_fil, defer_reduction=True)
            nondiff_dgms = top_opt.compute_diagram(include_inf_points=include_inf_points)
            self._top_opt = top_opt
            if n_threads is not None:
                # Stash on the under_fil so backward (with no direct constructor
                # access) can read it via ctx.fil._crit_sets_n_threads.
                try:
                    fil.under_fil._crit_sets_n_threads = int(n_threads)
                except Exception:
                    pass
            self._u_strategy = u_strategy
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
                "'dgm-loss' or 'crit-sets'"
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
                getattr(self, "_u_strategy", None),
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
    u_strategy=None,
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
        n_threads: passed to the underlying decomposition for "dgm-loss"
            and to the parallel V-only reduction for "crit-sets".
        u_strategy: which U-computation to use in the crit-sets
            backward. One of:
              - None (default): "auto".
              - "auto": production default. Currently resolves to
                row_partial with a partial-vs-full threshold dispatch
                (see ROW_PARTIAL_FULL_FALLBACK_THRESHOLD).
              - "row_partial": row-form partial U inversion. The
                algorithm behind 'auto'; pin this to bypass the
                threshold dispatch logic.
              - "legacy_in_band": in-band U built during reduction
                (clearing off, compute_u=true; serial reduction).
                Available as a cross-check / control strategy.

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
        u_strategy=u_strategy,
    )
