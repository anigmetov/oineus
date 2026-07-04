"""Differentiable mixup barcodes (torch and jax).

Differentiable variant of oineus.mixup_barcodes (Wagner, Arustamyan,
Wheeler, Bubenik, arXiv:2402.15058): the mixup triples (birth,
image_death, death) in every degree, and all the summary statistics
derived from them (total mixup, total / mean mixup percentage), are
differentiable with respect to the coordinates of both point clouds A
and B.

Gradient story: each finite mixup triple is identified by the sorted ids
of its (birth, image death, death) cells in the full filtration
K = VR(A u B), so the triples tensor is a pure gather
values_K[index_triples] into the values of the differentiable VR
filtration of the union -- exactly the dgm-loss mechanism of
oineus.diff.kicr_diagrams -- and gradients flow through the framework's
native autograd of the gather to the critical-edge distances and hence
to the coordinates of A and B. The statistics are smooth functions of
the triples (sums, ratios, means with a fixed bar count), so they are
differentiable wherever the underlying pairing is locally constant
(the generic situation, as with ordinary persistence).

Only finite triples are returned (mirroring oineus.diff.kicr_diagrams);
the statistics of the non-differentiable variant are computed over the
finite triples as well, so the two agree in value.

Note: like oineus.diff.vr_filtration, cell values are computed as
sqrt(dist^2 + eps), so degree-0 births are sqrt(eps) instead of 0 and
all values carry an O(eps) shift; reduce eps if this matters.
"""

import numpy as np

from .. import mixup as _mixup
from ._backend import infer_backend
from ._tensor_utils import tensor_to_real_numpy
from .vietoris_rips import vr_filtration as diff_vr_filtration

__all__ = ["DiffMixupBarcodes", "mixup_barcodes"]


def _gather_triples(values, index_triples, backend):
    """Tensor values[index_triples] of shape (n, 3) in the given framework;
    an empty index array yields an empty (0, 3) constant."""
    if backend == "torch":
        import torch
        if index_triples.size == 0:
            return torch.zeros((0, 3), dtype=values.dtype, device=values.device)
        return values[torch.from_numpy(index_triples).to(values.device)]
    if backend == "jax":
        import jax.numpy as jnp
        if index_triples.size == 0:
            return jnp.zeros((0, 3), dtype=values.dtype)
        return values[jnp.asarray(index_triples)]
    raise RuntimeError(f"unknown backend {backend!r}")


class DiffMixupBarcodes:
    """Differentiable mixup barcodes: dict-like dim -> (n, 3) tensor.

    Each row of in_dimension(dim) / self[dim] is a finite mixup triple
    (birth, image_death, death) with birth <= image_death <= death,
    differentiable with respect to the coordinates of A and B. The
    integer pairing behind each tensor (sorted ids in the union VR
    filtration) is available via index_triples_in_dimension.

    Statistics (per degree, differentiable scalars of the backend;
    a zero constant when the barcode in that degree is empty):

    * total_persistence(dim) = sum(death - birth)
    * total_mixup(dim) = sum(death - image_death)
    * total_mixup_percentage(dim) = sum((death - image_death) / (death - birth))
    * mean_mixup_percentage(dim) = their mean, in [0, 1]

    Essential bars are not represented (they carry zero mixup with the
    default truncation); the non-differentiable oineus.mixup_barcodes
    computes its statistics over the finite triples as well, so the
    statistics agree in value.
    """

    def __init__(self, diagrams, index_triples, backend, max_dim, ref_dtype, ref_device=None):
        self._diagrams = diagrams
        self._index_triples = index_triples
        self._backend = backend
        self.max_dim = max_dim
        # dtype (and device, for torch) of the zero constants returned by the
        # statistics of empty degrees
        self._ref_dtype = ref_dtype
        self._ref_device = ref_device

    def in_dimension(self, dim):
        """Finite mixup triples in the given degree as an (n, 3) tensor."""
        if dim not in self._diagrams:
            raise KeyError(f"no mixup barcode in dimension {dim}; "
                           f"available: {sorted(self._diagrams)}")
        return self._diagrams[dim]

    def __getitem__(self, dim):
        return self.in_dimension(dim)

    def __contains__(self, dim):
        return dim in self._diagrams

    def keys(self):
        return sorted(self._diagrams)

    def index_triples_in_dimension(self, dim):
        """Sorted ids in the union VR filtration of the (birth, image death,
        death) cells, an (n, 3) int64 numpy array (a copy)."""
        if dim not in self._index_triples:
            raise KeyError(f"no mixup index triples in dimension {dim}; "
                           f"available: {sorted(self._index_triples)}")
        return self._index_triples[dim].copy()

    def _zero(self):
        if self._backend == "torch":
            import torch
            return torch.zeros((), dtype=self._ref_dtype, device=self._ref_device)
        import jax.numpy as jnp
        return jnp.zeros((), dtype=self._ref_dtype)

    def total_persistence(self, dim):
        """Sum of death - birth over the finite triples (differentiable scalar)."""
        t = self.in_dimension(dim)
        return (t[:, 2] - t[:, 0]).sum() if len(t) else self._zero()

    def total_mixup(self, dim):
        """Sum of the mixups death - image_death (differentiable scalar)."""
        t = self.in_dimension(dim)
        return (t[:, 2] - t[:, 1]).sum() if len(t) else self._zero()

    def total_mixup_percentage(self, dim):
        """Sum of the per-bar mixup percentages (differentiable scalar)."""
        t = self.in_dimension(dim)
        if not len(t):
            return self._zero()
        return ((t[:, 2] - t[:, 1]) / (t[:, 2] - t[:, 0])).sum()

    def mean_mixup_percentage(self, dim):
        """Mean of the per-bar mixup percentages (differentiable scalar);
        a zero constant if the barcode is empty."""
        t = self.in_dimension(dim)
        if not len(t):
            return self._zero()
        return ((t[:, 2] - t[:, 1]) / (t[:, 2] - t[:, 0])).mean()

    def __repr__(self):
        sizes = {d: len(self._index_triples[d]) for d in sorted(self._index_triples)}
        return f"DiffMixupBarcodes(backend={self._backend!r}, max_dim={self.max_dim}, bars={sizes})"


def mixup_barcodes(A, B, max_dim=1, max_diameter=None, eps=1e-6, n_threads=1):
    """Differentiable mixup barcodes of the point cloud A included into A u B.

    Same construction as oineus.mixup_barcodes -- Vietoris-Rips filtrations
    L = VR(A) and K = VR(A u B) over a common range of scales, mixup triples
    (birth, image_death, death) per degree from the induced matching between
    the barcode of L and the image barcode of H(L) -> H(K) -- but A and B
    are torch tensors or jax arrays (same framework for both) and the
    returned triples and statistics are differentiable with respect to the
    coordinates of both A and B.

    Args:
        A: (n_A, d) tensor, the point cloud whose features are tracked.
        B: (n_B, d) tensor of the same framework, mixed into A; may be None
            or empty (all mixup sub-bars empty, statistics zero).
        max_dim: largest homological degree; simplices up to dimension
            max_dim + 1 are enumerated. Default 1.
        max_diameter: common truncation threshold for both filtrations;
            default: the enclosing radius of A (every finite bar of VR(A)
            and every image death lies below it).
        eps: smoothing constant of the differentiable VR values
            sqrt(dist^2 + eps) (see oineus.diff.vr_filtration).
        n_threads: parallelism of the underlying reductions.

    Returns:
        DiffMixupBarcodes with dict-like access to the (n, 3) triple tensors
        and differentiable statistics. Only finite triples are represented.

    JAX usage: an eager boundary, like oineus.diff.persistence_diagram --
    diagram sizes are data-dependent, so do not jit/vmap through this call;
    use it inside the function passed to jax.grad.
    """
    from .. import compute_kernel_image_cokernel_reduction, max_distance, vr_filtration, _oineus
    import eagerpy as epy

    backend = infer_backend(A)
    if backend is None:
        raise TypeError(
            "A must be a torch.Tensor or a jax array for differentiable "
            f"mixup barcodes, got {type(A).__name__}")
    if B is not None and len(B) and infer_backend(B) != backend:
        raise TypeError(
            f"A and B must belong to the same framework, got {backend!r} "
            f"and {infer_backend(B)!r}")
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")

    A_np = tensor_to_real_numpy(epy.astensor(A), dtype=np.float64)
    if A_np.ndim != 2:
        raise ValueError("A must be a 2D tensor of shape (n, d)")

    if A_np.shape[0] == 0:
        empty_t = {d: _gather_triples(A.reshape(-1), np.empty((0, 3), dtype=np.int64), backend)
                   for d in range(max_dim + 1)}
        empty_i = {d: np.empty((0, 3), dtype=np.int64) for d in range(max_dim + 1)}
        device = A.device if backend == "torch" else None
        return DiffMixupBarcodes(empty_t, empty_i, backend, max_dim, A.dtype, device)

    if max_diameter is None:
        max_diameter = float(max_distance(A_np)) if len(A_np) >= 2 else 0.0

    have_B = B is not None and len(B) > 0
    if have_B:
        if backend == "torch":
            import torch
            union = torch.cat([A, B], dim=0)
        else:
            import jax.numpy as jnp
            union = jnp.concatenate([A, B], axis=0)
    else:
        union = A

    # differentiable filtration of the union; fat cells (see oineus.mixup:
    # packed uids depend on the point count, so packed K and L would not
    # share uids)
    K = diff_vr_filtration(union, max_dim=max_dim + 1, max_diameter=max_diameter,
                           eps=eps, packed=False, n_threads=n_threads)
    L = vr_filtration(A_np, max_dim=max_dim + 1, max_diameter=max_diameter,
                      packed=False, n_threads=n_threads)

    params = _oineus.KICRParams()
    params.kernel = False
    params.image = True
    params.cokernel = False
    params.codomain = False
    params.include_zero_persistence = False
    params.n_threads = max(1, int(n_threads))

    kicr = compute_kernel_image_cokernel_reduction(K.under_fil, L, params)
    triples = _mixup.compute_mixup_triples(kicr, K.under_fil, L, max_dim)

    diagrams, index_triples = {}, {}
    for dim, (_finite_vals, finite_idx, _essential) in triples.items():
        diagrams[dim] = _gather_triples(K.values, finite_idx, backend)
        index_triples[dim] = finite_idx

    device = K.values.device if backend == "torch" else None
    return DiffMixupBarcodes(diagrams, index_triples, backend, max_dim,
                             K.values.dtype, device)
