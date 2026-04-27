from .. import _oineus
import torch
import numpy as np

from .diff_filtration import DiffFiltration


class PersistenceDiagramHelper(torch.autograd.Function):
    """
    Autograd function for extracting a single dimension's diagram.

    Forward: subscripts fil.values at birth/death indices
    Backward: accumulates gradients at those indices (dgm-loss)
              or expands to critical sets (crit-sets)
    """

    @staticmethod
    def forward(ctx, fil_values, fil, dcmp_hom, dcmp_coh, dgms, dim, include_inf_points,
                gradient_method, lr, conflict_strategy):
        """
        Extract diagram for dimension `dim` as a differentiable tensor.
        """
        # Get index diagram (list of IndexDiagramPoint)
        index_dgm = dgms.index_diagram_in_dimension(dim, as_numpy=True).astype(np.int64)
        index_dgm = torch.from_numpy(index_dgm).to(fil_values.device)

        fil_len = fil_values.shape[0]

        ctx.gradient_method = gradient_method
        ctx.fil_len = fil_len
        ctx.include_inf_points = include_inf_points
        ctx.lr = lr
        ctx.conflict_strategy = conflict_strategy
        ctx.dim = dim

        n_total = len(index_dgm)

        if n_total == 0:
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

        # Save for backward
        ctx.save_for_backward(index_dgm)

        return diagram

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: propagate gradients to fil_values.
        """
        if ctx.gradient_method == "dgm-loss":
            # w.r.t. fil_values, fil, dcmp_hom, dcmp_coh, dgms, dim,  include_inf_points, gradient_method, lr, conflict_strategy
            fil_len = ctx.fil_len
            index_dgm = ctx.saved_tensors[0]
            grad_vals = torch.zeros(fil_len, dtype=grad_output.dtype, device=grad_output.device)
            grad_vals[index_dgm.flatten()] = grad_output.flatten()
            return grad_vals, None, None, None, None, None, None, None, None, None
        else:
            raise RuntimeError("Gradient method not implemented yet.")


class PersistenceDiagrams:
    """
    Container for differentiable persistence diagrams in all dimensions.
    Provides access to diagrams in each dimension as differentiable tensors.

    Usage:
        dgms = persistence_diagram(fil, dualize=True)
        dgm1 = dgms[1]  # H1 diagram as tensor (N, 2)
        loss = dgm1[:, 1].sum()
        loss.backward()
    """

    def __init__(self, fil: DiffFiltration, dualize: bool, include_inf_points: bool,
                 gradient_method: str, lr: float, conflict_strategy: str, rp = None,
                 n_threads = None):
        if not isinstance(fil.values, torch.Tensor):
            raise TypeError("fil.values must be a torch.Tensor for differentiable diagrams")

        if rp is None:
            rp = _oineus.ReductionParams()

        self._fil = fil
        self._dualize = dualize
        if n_threads is None:
            dcmp = _oineus.Decomposition(fil.under_fil, dualize=dualize)
        else:
            dcmp = _oineus.Decomposition(fil.under_fil, dualize=dualize, n_threads=n_threads)
        dcmp.reduce(rp)
        if dualize:
            self._dcmp_coh, self._dcmp_hom = dcmp, None
        else:
            self._dcmp_coh, self._dcmp_hom = None, dcmp
        self._include_inf_points = include_inf_points
        self._gradient_method = gradient_method
        self._lr = lr
        self._conflict_strategy = conflict_strategy

        nondiff_dgms = dcmp.diagram(fil.under_fil, include_inf_points=include_inf_points)

        self._diagrams = { dim : PersistenceDiagramHelper.apply(
                self._fil.values,
                self._fil.under_fil,
                self._dcmp_hom,
                self._dcmp_coh,
                nondiff_dgms,
                dim,
                self._include_inf_points,
                self._gradient_method,
                self._lr,
                self._conflict_strategy
            )
        for dim in range(self._fil.max_dim()) }


    def __getitem__(self, dim: int) -> torch.Tensor:
        """Get diagram in dimension dim."""
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
        """Alias for __getitem__."""
        return self[dim]

    @property
    def max_dim(self) -> int:
        """Maximum dimension available."""
        return max(self._diagrams.keys())


def persistence_diagram(
    fil: DiffFiltration,
    dualize: bool = False,
    include_inf_points: bool = False,
    gradient_method: str = "dgm-loss",
    lr: float = 1.0,
    conflict_strategy: str = "avg",
    n_threads = None,
) -> PersistenceDiagrams:
    """
    Compute differentiable persistence diagrams from a DiffFiltration.

    Efficiently computes the decomposition once and returns diagrams for
    all dimensions. The returned object can be indexed by dimension.

    Args:
        fil: DiffFiltration with differentiable `values` tensor
        dualize: If True, compute cohomology (default). If False, homology.
        include_inf_points: If True, include infinite points in diagrams.
                           Infinite deaths are represented as float('inf').
        gradient_method: "dgm-loss" (gradient to critical simplices only)
                        or "crit-sets" (gradient to all simplices in critical sets)
        lr: Learning rate for crit-sets target computation (default: 1.0)
        conflict_strategy: Conflict resolution for crit-sets: "avg", "max", "sum"

    Returns:
        PersistenceDiagrams: dict-like object mapping dimension -> Tensor (N, 2)
        Each tensor has birth values in column 0, death values in column 1.
        Gradients flow back to fil.values automatically.

    Example:
        # Create VR filtration from points
        pts = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]], requires_grad=True)
        fil = oin.diff.vr_filtration(pts, max_dim=2)

        # Get all diagrams
        dgms = oin.diff.persistence_diagram(fil, dualize=True)

        # Access specific dimension
        dgm1 = dgms[1]  # H1 diagram

        # Compute loss and backpropagate
        loss = (dgm1[:, 1] - 2.0).pow(2).sum()
        loss.backward()
        print(pts.grad)
    """
    return PersistenceDiagrams(
        fil=fil,
        dualize=dualize,
        include_inf_points=include_inf_points,
        gradient_method=gradient_method,
        lr=lr,
        conflict_strategy=conflict_strategy,
        n_threads=n_threads,
    )
