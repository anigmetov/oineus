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
        ctx.conflict_strategy = conflict_strategy

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

        if ctx.gradient_method != "crit-sets":
            raise RuntimeError(f"unknown gradient_method {ctx.gradient_method!r}")

        # Crit-sets path. We restrict to finite pairs; rows of grad_output match
        # the layout produced by forward (finite pairs first when
        # include_inf_points is True). For include_inf_points=False, every row
        # is finite by construction.
        n_total = index_dgm.shape[0]
        device = grad_output.device

        if n_total == 0:
            return torch.zeros(fil_len, dtype=grad_output.dtype, device=device), None, None, None, None, None, None, None, None

        if ctx.include_inf_points:
            fin_mask = (index_dgm[:, 0] >= 0) & (index_dgm[:, 0] < fil_len) & (index_dgm[:, 1] >= 0) & (index_dgm[:, 1] < fil_len)
            n_fin = int(fin_mask.sum().item())
            fin_idx_dgm = index_dgm[fin_mask]
            fin_grad = grad_output[:n_fin]
        else:
            fin_idx_dgm = index_dgm
            fin_grad = grad_output

        grad_vals = torch.zeros(fil_len, dtype=grad_output.dtype, device=device)

        if fin_idx_dgm.shape[0] == 0:
            return grad_vals, None, None, None, None, None, None, None, None

        b_idx = fin_idx_dgm[:, 0]
        d_idx = fin_idx_dgm[:, 1]
        b_cur = fil_values[b_idx]
        d_cur = fil_values[d_idx]
        step = ctx.step_size
        b_tgt = b_cur - step * fin_grad[:, 0]
        d_tgt = d_cur - step * fin_grad[:, 1]

        # Drop no-op moves and concatenate (b_idx, b_tgt) with (d_idx, d_tgt).
        b_move = b_tgt != b_cur
        d_move = d_tgt != d_cur
        flat_idx = torch.cat([b_idx[b_move], d_idx[d_move]])
        flat_tgt = torch.cat([b_tgt[b_move], d_tgt[d_move]])

        if flat_idx.numel() == 0:
            return grad_vals, None, None, None, None, None, None, None, None

        flat_idx_np = flat_idx.detach().cpu().numpy().tolist()
        flat_tgt_np = flat_tgt.detach().cpu().to(torch.float64).numpy().tolist()

        top_opt = ctx.top_opt
        crit_sets = top_opt.singletons(flat_idx_np, flat_tgt_np)
        strategy = _resolve_strategy(ctx.conflict_strategy)
        indvals = top_opt.combine_loss(crit_sets, strategy)
        indices = list(indvals[0])
        targets = list(indvals[1])

        if not indices:
            return grad_vals, None, None, None, None, None, None, None, None

        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        tgt_t = torch.tensor(targets, dtype=grad_output.dtype, device=device)

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

    def __init__(self, fil: DiffFiltration, dualize: bool, include_inf_points: bool,
                 gradient_method: str, step_size: float, conflict_strategy,
                 rp=None, n_threads=None):
        if not isinstance(fil.values, torch.Tensor):
            raise TypeError("fil.values must be a torch.Tensor for differentiable diagrams")

        if rp is None:
            rp = _oineus.ReductionParams()

        self._fil = fil
        self._dualize = dualize
        self._include_inf_points = include_inf_points
        self._gradient_method = gradient_method
        self._step_size = step_size
        self._conflict_strategy = conflict_strategy

        if gradient_method == "crit-sets":
            top_opt = _oineus.TopologyOptimizer(fil.under_fil)
            top_opt.reduce_all()
            nondiff_dgms = top_opt.compute_diagram(include_inf_points=include_inf_points)
            self._top_opt = top_opt
        elif gradient_method == "dgm-loss":
            kwargs = {"dualize": dualize}
            if n_threads is not None:
                kwargs["n_threads"] = n_threads
            dcmp = _oineus.Decomposition(fil.under_fil, **kwargs)
            dcmp.reduce(rp)
            nondiff_dgms = dcmp.diagram(fil.under_fil, include_inf_points=include_inf_points)
            self._top_opt = dcmp
        else:
            raise ValueError(
                f"unknown gradient_method {gradient_method!r}; expected 'dgm-loss' or 'crit-sets'"
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
    dualize: bool = False,
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
        dualize: cohomology if True, homology if False. Used only for
            "dgm-loss"; "crit-sets" always builds both decompositions.
        include_inf_points: if True, include infinite points (deaths set
            to float('inf')); only finite pairs receive crit-sets gradient.
        gradient_method: "dgm-loss" or "crit-sets".
        step_size: scales grad_output to a target diagram (target =
            current - step_size * grad_output) for the "crit-sets" pass.
            Ignored for "dgm-loss".
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
