"""Differentiable persistence diagrams.

Works with torch tensors and jax arrays alike: the backend is detected
from ``fil.values`` and the diagrams come back as that framework's
tensors, with gradients flowing to the filtration values (and through
them to whatever produced the values -- points, images, ...). The
4-matrix bookkeeping lives in the framework-neutral pd_core; pd_torch
and pd_jax are thin adapters over it.

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

JAX notes: oineus.diff is an eager boundary -- diagram sizes are
data-dependent, so do not jit/vmap through the filtration/diagram
calls (jit the network and the loss around them instead). Without
``jax.config.update("jax_enable_x64", True)`` jax arrays are float32
and route to the float32 oineus backend when it is compiled in; with
x64 enabled float64 arrays route to the default float64 backend.

Phase 1 of the refactor only supports `include_inf_points=False`.
Phase 2 will expose a split index-diagram from C++ for the
inf-points case so it can be handled without per-entry validity
masks.
"""

from .diff_filtration import DiffFiltration
from ._backend import infer_backend, concrete_numpy
from . import pd_core


class PersistenceDiagrams:
    """Container for differentiable persistence diagrams in all dimensions.

    Usage:
        dgms = persistence_diagram(fil)
        dgm1 = dgms[1]                # H1 diagram as tensor (N, 2)
        loss = (dgm1[:, 1] - dgm1[:, 0]).pow(2).sum()
        loss.backward()

    The diagrams are torch tensors for a torch-valued filtration and jax
    arrays for a jax-valued one (use jax.grad on a function that builds
    the filtration and the loss).
    """

    def __init__(self, fil: DiffFiltration, *, dualize, include_inf_points,
                 gradient_method, step_size, conflict_strategy,
                 n_threads, u_strategy, dims_to_backprop):
        backend = infer_backend(fil.values)
        if backend is None:
            raise TypeError(
                "fil.values must be a torch.Tensor or a jax array for "
                f"differentiable diagrams, got {type(fil.values).__name__}")

        if include_inf_points:
            raise NotImplementedError(
                "include_inf_points=True is deferred to Phase 2 of the "
                "differentiable-diagram refactor (will need a split "
                "index-diagram return type from C++ to avoid per-entry "
                "validity masks). For now, request finite points only.")

        fwd = pd_core.pd_forward(
            fil.under_fil,
            concrete_numpy(fil.values),
            dualize=dualize,
            method=gradient_method,
            dims_to_backprop=dims_to_backprop,
            n_threads=n_threads,
            u_strategy=u_strategy,
            conflict_strategy=conflict_strategy,
            step_size=step_size,
            max_dim=fil.max_dim,
        )

        self._fil = fil
        self._top_opt = fwd.top_opt
        self._dualize = fwd.dualize
        self._gradient_method = gradient_method

        if backend == "torch":
            from . import pd_torch
            diagram_fn = pd_torch.torch_diagram
        else:
            from . import pd_jax
            diagram_fn = pd_jax.jax_diagram

        self._diagrams = {
            dim: diagram_fn(fil.values, fwd, dim)
            for dim in range(fil.max_dim)
        }

    def __getitem__(self, dim: int):
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

    def in_dimension(self, dim: int):
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
        fil: DiffFiltration with differentiable `values` tensor (torch
            or jax; the backend is auto-detected and the diagrams are
            returned as that framework's tensors).
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
        PersistenceDiagrams: dict-like, dim -> tensor (N, 2). Gradients
        flow back to fil.values.

    JAX usage: build the filtration and the diagram inside the function
    you differentiate (jax.grad); do not jit through this call -- diagram
    sizes are data-dependent, so oineus.diff is an eager boundary. jax
    arrays are float32 unless jax_enable_x64 is set; float32 inputs use
    the float32 oineus backend when compiled in, float64 the default one.
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
