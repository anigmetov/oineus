"""Differentiable kernel/image/cokernel (KICR) persistence diagrams.

For an inclusion of filtrations L -> K (L a subcomplex of K carrying the
same values on shared cells), the kernel, image and cokernel persistence
diagrams of the induced map on homology (Cohen-Steiner, Edelsbrunner,
Harer, Morozov, "Persistent Homology for Kernels, Images, and Cokernels",
SODA 2009) are computed by the C++ KerImCokReduced and returned as
framework tensors -- torch or jax, detected from K.values.

Gradient story: every birth/death cell of every family is identified by
its sorted id in the FULL filtration K (kernel.h reads all diagram values
through fil_K, including kernel death cells that live in L), so each
diagram is a pure gather values_K[index_pairs] and gradients flow to
K.values through the framework's native autograd of the gather (its VJP
is the scatter-add) -- exactly the dgm-loss method of
persistence_diagram. There is no crit-sets machinery for KICR. If L is
passed as a DiffFiltration, only its underlying filtration is used; its
values tensor never receives gradients -- route the values through K.

Only finite points are returned (include_inf_points=False), mirroring
Phase 1 of the differentiable-diagram refactor; the non-diff
kernel_diagrams()/image_diagrams()/cokernel_diagrams() include the
points at infinity.
"""

from ._backend import infer_backend
from .diff_filtration import DiffFiltration
from . import pd_core


def gather_diagram(values, index_dgm, backend):
    """Tensor values[index_dgm] of shape (n_d, 2) in the given framework.

    index_dgm is an (n_d, 2) int64 numpy array of sorted ids into values.
    Empty index diagrams produce an empty (0, 2) constant of the right
    dtype (and device, for torch). The native VJP of the gather is the
    scatter-add, so no custom backward is needed.
    """
    if backend == "torch":
        import torch
        if index_dgm.size == 0:
            return torch.zeros((0, 2), dtype=values.dtype, device=values.device)
        return values[torch.from_numpy(index_dgm).to(values.device)]
    if backend == "jax":
        import jax.numpy as jnp
        if index_dgm.size == 0:
            return jnp.zeros((0, 2), dtype=values.dtype)
        return values[jnp.asarray(index_dgm)]
    raise RuntimeError(f"unknown backend {backend!r}")


class KICRFamilyDiagrams:
    """Diagrams of one family (kernel, image, or cokernel).

    Dict-like: dim -> (n_d, 2) tensor of (birth, death), differentiable
    with respect to the values of the full filtration K. The integer
    pairing behind each tensor is available via
    index_diagram_in_dimension (sorted ids in K, finite points only).
    """

    def __init__(self, family, diagrams, index_dgms):
        self.family = family
        self._diagrams = diagrams
        self._index_dgms = index_dgms

    def __getitem__(self, dim: int):
        if dim not in self._diagrams:
            raise KeyError(
                f"No {self.family} diagram for dimension {dim}. "
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

    def index_diagram_in_dimension(self, dim: int):
        """Integer pairing in dimension dim: an (n_d, 2) int64 numpy array
        of (birth, death) sorted ids in the full filtration K (finite
        points only)."""
        if dim not in self._index_dgms:
            raise KeyError(
                f"No {self.family} index diagram for dimension {dim}. "
                f"Available: {list(self._index_dgms.keys())}")
        return self._index_dgms[dim]

    @property
    def max_dim(self) -> int:
        return max(self._diagrams.keys())

    def __repr__(self):
        sizes = {d: len(self._index_dgms[d]) for d in sorted(self._index_dgms)}
        return f"KICRFamilyDiagrams(family={self.family!r}, sizes={sizes})"


class KICRDiagrams:
    """Differentiable kernel/image/cokernel diagrams of an inclusion L -> K.

    Usage:
        K = oineus.diff.vr_filtration(pts, max_dim=2, max_diameter=2.0)
        L = K.under_fil.without_cells(ids_to_remove)
        dgms = oineus.diff.kicr_diagrams(K, L)
        ker1 = dgms.kernel[1]                     # (n, 2) tensor
        loss = ((ker1[:, 1] - ker1[:, 0]) ** 2).sum()
        loss.backward()                           # gradient lands on K.values

    The .kernel / .image / .cokernel properties are KICRFamilyDiagrams;
    accessing a family that was not computed raises RuntimeError. The
    live C++ KerImCokReduced is exposed as .reduction.
    """

    def __init__(self, K: DiffFiltration, L, *, kernel=True, image=True,
                 cokernel=True, include_zero_persistence=False,
                 include_inf_points=False, n_threads=1):
        if not isinstance(K, DiffFiltration):
            raise TypeError(
                "K must be a DiffFiltration (wrapping a differentiable "
                f"values tensor), got {type(K).__name__}")
        backend = infer_backend(K.values)
        if backend is None:
            raise TypeError(
                "K.values must be a torch.Tensor or a jax array for "
                f"differentiable KICR diagrams, got {type(K.values).__name__}")

        if include_inf_points:
            raise NotImplementedError(
                "include_inf_points=True is deferred to Phase 2 of the "
                "differentiable-diagram refactor (needs a split "
                "index-diagram return type to avoid per-entry validity "
                "masks). For now, request finite points only; the non-diff "
                "KICR diagrams do include points at infinity.")

        under_L = L.under_fil if isinstance(L, DiffFiltration) else L

        fwd = pd_core.kicr_forward(
            K.under_fil, under_L,
            kernel=kernel, image=image, cokernel=cokernel,
            include_zero_persistence=include_zero_persistence,
            n_threads=n_threads,
        )

        self._fil = K
        self.reduction = fwd.kicr
        self._families = {}
        for family in fwd.families:
            by_dim = fwd.index_dgms[family]
            diagrams = {dim: gather_diagram(K.values, idx, backend)
                        for dim, idx in by_dim.items()}
            self._families[family] = KICRFamilyDiagrams(family, diagrams, by_dim)

    def _family(self, name):
        fam = self._families.get(name)
        if fam is None:
            raise RuntimeError(
                f"{name} diagrams were not computed because {name}=False "
                "was passed to kicr_diagrams")
        return fam

    @property
    def kernel(self) -> KICRFamilyDiagrams:
        return self._family("kernel")

    @property
    def image(self) -> KICRFamilyDiagrams:
        return self._family("image")

    @property
    def cokernel(self) -> KICRFamilyDiagrams:
        return self._family("cokernel")

    @property
    def families(self) -> tuple:
        """Names of the families that were computed."""
        return tuple(self._families.keys())

    def __repr__(self):
        inner = ", ".join(repr(self._families[f]) for f in self._families)
        return f"KICRDiagrams({inner})"


def kicr_diagrams(K: DiffFiltration, L, *, kernel=True, image=True,
                  cokernel=True, include_zero_persistence=False,
                  include_inf_points=False, n_threads=1) -> KICRDiagrams:
    """Differentiable kernel/image/cokernel persistence diagrams of the
    inclusion L -> K.

    Args:
        K: DiffFiltration of the full complex; its values tensor (torch
            or jax, backend auto-detected) receives the gradients.
        L: subcomplex of K -- a plain oineus filtration, typically
            K.under_fil.without_cells(...) or another subfiltration
            construction, or a DiffFiltration (only its underlying
            filtration is used). L must carry the same values as K on
            shared cells (the CEHM setting g = f restricted to L); like
            the non-diff API, this is not checked. Gradients flow to
            K.values only.
        kernel, image, cokernel: which families to compute. At least one
            must be True; accessing a disabled family on the result
            raises RuntimeError.
        include_zero_persistence: keep pairs with birth == death. Default
            False, mirroring the non-diff KICR diagrams.
        include_inf_points: only False is supported (Phase 1 of the
            differentiable-diagram refactor); True raises
            NotImplementedError.
        n_threads: parallelism of the C++ reductions (propagated to all
            KICR stages).

    Returns:
        KICRDiagrams with .kernel / .image / .cokernel, each dict-like
        dim -> (n_d, 2) tensor of finite (birth, death) points. Gradients
        flow back to K.values (and through them to whatever produced the
        values). Only diagram-indexing gradients are supported (the
        dgm-loss method); there is no crit-sets machinery for KICR.

    JAX usage: as with persistence_diagram, this is an eager boundary --
    diagram sizes are data-dependent, so do not jit/vmap through this
    call; use it inside the function passed to jax.grad.
    """
    return KICRDiagrams(
        K, L,
        kernel=kernel, image=image, cokernel=cokernel,
        include_zero_persistence=include_zero_persistence,
        include_inf_points=include_inf_points,
        n_threads=n_threads,
    )
