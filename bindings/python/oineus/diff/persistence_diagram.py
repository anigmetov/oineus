"""Differentiable persistence diagrams.

Two `gradient_method` options:

- "dgm-loss":  gradient flows only through the (birth_simplex,
               death_simplex) pair defining each diagram point. The
               forward path reduces one decomposition (hom or coh,
               chosen by `dualize`) with the cheapest recipe -
               parallel + clearing, R only - and the backward is a
               scatter into the fil_values gradient.

- "crit-sets": gradient propagates through the full critical set of
               each moved pair, with conflicts resolved by the
               selected `conflict_strategy`. The forward reduces one
               side with parallel + clearing + V + restore_ELZ in
               `dims_to_backprop` so the backward can recover U on
               demand without re-reducing. The other decomposition
               is reduced lazily in backward only if
               `determine_needed_matrices` says we need it.

Phase 1 of the refactor only supports `include_inf_points=False`.
Phase 2 will expose a split index-diagram from C++ for the
inf-points case so it can be handled without per-entry validity
masks.
"""

from .. import _oineus
import torch
import numpy as np

from .diff_filtration import DiffFiltration
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


def _resolve_u_strategy(u_strategy):
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


def determine_needed_matrices(dgm_grad, negate: bool):
    """Return `(v_hom, u_hom, v_coh, u_coh)`: which of the four
    matrices the crit-sets backward needs given a diagram gradient.

    Single torch op over `dgm_grad`; cheap and GPU-friendly.

    A positive sign on birth/death pushes the value down (we minimize
    the loss), a negative sign pushes it up. Mapping the sign to the
    move direction and the move direction to the matrix (paper:
    decrease_birth -> U_coh, increase_birth -> V_coh,
    increase_death -> U_hom, decrease_death -> V_hom) gives the
    layout below. `negate=True` flips value-direction vs
    filtration-direction, so the matrix assignment swaps accordingly.
    """
    if dgm_grad.numel() == 0:
        return False, False, False, False
    mn, mx = torch.aminmax(dgm_grad, dim=0)
    flags = torch.cat([mx > 0, mn < 0]).tolist()
    if negate:
        v_coh, u_hom, u_coh, v_hom = flags
    else:
        u_coh, v_hom, v_coh, u_hom = flags
    return v_hom, u_hom, v_coh, u_coh


def _select_u_moves(idx_t, cur_t, tgt_t, move_t, *, side, negate):
    """Pick the rows + bounds the U-side walker on `side` will read.

    Filters moves to those in the U-needing direction:
      hom side, non-negate: increase_death (tgt > cur).
      coh side, non-negate: decrease_birth (tgt < cur).
      negate flips both.
    Returns (rows_fil_idx_list, bounds_list) ready for
    ensure_has_u_hom / ensure_has_u_coh.
    """
    if side == "hom":
        u_dir_mask = (tgt_t < cur_t) if negate else (tgt_t > cur_t)
    else:
        u_dir_mask = (tgt_t > cur_t) if negate else (tgt_t < cur_t)

    sel = move_t & u_dir_mask
    if not bool(sel.any().item()):
        return [], []
    rows = idx_t[sel].detach().cpu().numpy().astype(np.uintp).tolist()
    bounds = tgt_t[sel].detach().cpu().numpy().tolist()
    return rows, bounds


def _forward_reduce(top_opt, dualize):
    """Reduce the chosen side with the recipe baked into the
    optimizer (recipe was decided at construction time)."""
    if dualize:
        top_opt.ensure_coh_reduced()
        return top_opt.cohomology_decomposition_ref()
    top_opt.ensure_hom_reduced()
    return top_opt.homology_decomposition_ref()


def _backward_dgm_loss(ctx, grad_output):
    index_dgm, fil_values = ctx.saved_tensors
    grad_vals = torch.zeros_like(fil_values)
    grad_vals.scatter_add_(0, index_dgm[:, 0], grad_output[:, 0])
    grad_vals.scatter_add_(0, index_dgm[:, 1], grad_output[:, 1])
    return (grad_vals,) + (None,) * 7


def _backward_crit_sets(ctx, grad_output):
    index_dgm, fil_values = ctx.saved_tensors
    top_opt = ctx.top_opt
    negate = ctx.negate

    b_idx = index_dgm[:, 0]
    d_idx = index_dgm[:, 1]
    b_cur = fil_values[b_idx]
    d_cur = fil_values[d_idx]
    b_tgt = b_cur - ctx.step_size * grad_output[:, 0]
    d_tgt = d_cur - ctx.step_size * grad_output[:, 1]
    b_move = b_tgt != b_cur
    d_move = d_tgt != d_cur

    v_hom, u_hom, v_coh, u_coh = determine_needed_matrices(grad_output, negate)

    if v_hom or u_hom:
        top_opt.ensure_hom_reduced()
    if v_coh or u_coh:
        top_opt.ensure_coh_reduced()

    if u_hom:
        rows, bounds = _select_u_moves(d_idx, d_cur, d_tgt, d_move,
                                       side="hom", negate=negate)
        top_opt.ensure_has_u_hom(ctx.dim, rows, bounds)
    if u_coh:
        rows, bounds = _select_u_moves(b_idx, b_cur, b_tgt, b_move,
                                       side="coh", negate=negate)
        top_opt.ensure_has_u_coh(ctx.dim, rows, bounds)

    flat_idx = torch.cat([b_idx[b_move], d_idx[d_move]])
    flat_tgt = torch.cat([b_tgt[b_move], d_tgt[d_move]])

    grad_vals = torch.zeros_like(fil_values)
    if flat_idx.numel() == 0:
        return (grad_vals,) + (None,) * 7

    flat_idx_np = flat_idx.detach().cpu().numpy().astype(np.uintp).tolist()
    flat_tgt_np = flat_tgt.detach().cpu().numpy().tolist()

    # crit_sets_apply handles the dispatch reduction (ensure_hom_reduced)
    # internally and raises if the optimizer is dgm-loss only.
    indvals = top_opt.crit_sets_apply(flat_idx_np, flat_tgt_np, ctx.strategy)
    out_idx_np = np.asarray(indvals.indices_array(), copy=True)
    out_tgt_np = np.asarray(indvals.values_array(), copy=True)
    if out_idx_np.size == 0:
        return (grad_vals,) + (None,) * 7

    idx_t = torch.from_numpy(out_idx_np.astype(np.int64)).to(device=fil_values.device)
    tgt_t = torch.from_numpy(out_tgt_np).to(dtype=fil_values.dtype,
                                            device=fil_values.device)
    if ctx.strategy == _oineus.ConflictStrategy.Sum:
        contrib = fil_values[idx_t] - tgt_t
        grad_vals.scatter_add_(0, idx_t, contrib)
    else:
        grad_vals[idx_t] = fil_values[idx_t] - tgt_t

    return (grad_vals,) + (None,) * 7


class _PDHelper(torch.autograd.Function):
    """One autograd Function per (dim, gradient_method) pair. Forward
    subscripts fil.values at birth/death indices; backward dispatches
    to dgm-loss (scatter) or crit-sets."""

    @staticmethod
    def forward(ctx, fil_values, top_opt, nondiff_dgms, dim,
                gradient_method, step_size, strategy, negate):
        index_dgm = nondiff_dgms.index_diagram_in_dimension(
            dim, as_numpy=True).astype(np.int64)
        index_dgm = torch.from_numpy(index_dgm).to(fil_values.device)

        if index_dgm.numel() == 0:
            diagram = torch.zeros((0, 2), dtype=fil_values.dtype,
                                  device=fil_values.device)
        else:
            diagram = fil_values[index_dgm]

        ctx.save_for_backward(index_dgm, fil_values)
        ctx.top_opt = top_opt
        ctx.dim = dim
        ctx.gradient_method = gradient_method
        ctx.step_size = step_size
        ctx.strategy = strategy
        ctx.negate = negate
        return diagram

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.gradient_method == "dgm-loss":
            return _backward_dgm_loss(ctx, grad_output)
        if ctx.gradient_method == "crit-sets":
            return _backward_crit_sets(ctx, grad_output)
        raise RuntimeError(f"Unknown gradient method: {ctx.gradient_method}")


class PersistenceDiagrams:
    """Container for differentiable persistence diagrams in all dimensions.

    Usage:
        dgms = persistence_diagram(fil)
        dgm1 = dgms[1]                # H1 diagram as tensor (N, 2)
        loss = (dgm1[:, 1] - dgm1[:, 0]).pow(2).sum()
        loss.backward()
    """

    def __init__(self, fil: DiffFiltration, *, dualize, include_inf_points,
                 gradient_method, step_size, conflict_strategy,
                 n_threads, u_strategy, dims_to_backprop):
        if not isinstance(fil.values, torch.Tensor):
            raise TypeError("fil.values must be a torch.Tensor for "
                            "differentiable diagrams")

        if include_inf_points:
            raise NotImplementedError(
                "include_inf_points=True is deferred to Phase 2 of the "
                "differentiable-diagram refactor (will need a split "
                "index-diagram return type from C++ to avoid per-entry "
                "validity masks). For now, request finite points only.")

        if dualize is None:
            dualize = default_dualize_for_filtration(fil.under_fil)

        if dims_to_backprop is None:
            # Cover all simplex dims so partial-U is admissible
            # everywhere. For H_k pairs the birth simplex has dim k
            # and the death simplex has dim k+1, so we need
            # range(max_dim + 1).
            dims_to_backprop = list(range(fil.max_dim + 1))

        n_threads = max(1, int(n_threads) if n_threads is not None else 1)
        strategy = _resolve_strategy(conflict_strategy)
        u_strategy_enum = _resolve_u_strategy(u_strategy)
        negate = bool(fil.negate)
        with_crit_sets = gradient_method == "crit-sets"

        top_opt = TopologyOptimizer(
            fil,
            with_crit_sets=with_crit_sets,
            dims_to_restore_elz=dims_to_backprop,
            n_threads=n_threads,
            u_strategy=u_strategy_enum,
        )
        decmp = _forward_reduce(top_opt, dualize)
        nondiff_dgms = decmp.diagram(fil.under_fil,
                                     include_inf_points=False)

        self._fil = fil
        self._top_opt = top_opt
        self._dualize = dualize
        self._gradient_method = gradient_method

        self._diagrams = {
            dim: _PDHelper.apply(
                fil.values, top_opt, nondiff_dgms, dim,
                gradient_method, step_size, strategy, negate,
            )
            for dim in range(fil.max_dim)
        }

    def __getitem__(self, dim: int) -> torch.Tensor:
        if dim not in self._diagrams:
            raise KeyError(
                f"No diagram for dimension {dim}. "
                f"Available: {list(self._diagrams.keys())}")
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
    dims_to_backprop=None,
) -> PersistenceDiagrams:
    """Compute differentiable persistence diagrams from a DiffFiltration.

    Args:
        fil: DiffFiltration with differentiable `values` tensor.
        dualize: cohomology if True, homology if False. None (default)
            uses the FiltrationKind reduction policy. Currently this picks
            cohomology for VR and homology otherwise.
        include_inf_points: Phase 1 only supports False. Setting True
            raises NotImplementedError.
        gradient_method: "dgm-loss" or "crit-sets".
        step_size: scales grad_output to a target diagram
            (target = current - step_size * grad_output) for crit-sets.
            Ignored for dgm-loss.
        conflict_strategy: "avg", "max", "sum", or "fca", or any
            _oineus.ConflictStrategy. Used only for crit-sets.
        n_threads: parallelism for the forward reduction and the
            partial-U pass in backward.
        u_strategy: "auto" (default), "row_partial", or
            "legacy_in_band", or any _oineus.UStrategy. Used only for
            crit-sets.
        dims_to_backprop: list of geometric dims to restore ELZ in
            during the forward reduction. None defaults to all dims
            of the filtration. Used only for crit-sets.

    Returns:
        PersistenceDiagrams: dict-like, dim -> Tensor (N, 2). Gradients
        flow back to fil.values.
    """
    return PersistenceDiagrams(
        fil,
        dualize=dualize,
        include_inf_points=include_inf_points,
        gradient_method=gradient_method,
        step_size=step_size,
        conflict_strategy=conflict_strategy,
        n_threads=n_threads,
        u_strategy=u_strategy,
        dims_to_backprop=dims_to_backprop,
    )
