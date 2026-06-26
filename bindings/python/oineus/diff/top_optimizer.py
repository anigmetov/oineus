from typing import List, Optional

from .. import _oineus
from .. import _OPT_CLASS_BY_FIL_TYPE
from .diff_filtration import DiffFiltration
from ._reduction_policy import default_dualize_for_filtration


def _opt_class_for_filtration(fil):
    """Pick the C++ TopologyOptimizer instantiation for this cell type."""
    if isinstance(fil, DiffFiltration):
        fil = fil.under_fil
    cls = _OPT_CLASS_BY_FIL_TYPE.get(type(fil))
    if cls is None:
        raise RuntimeError(
            f"unknown filtration type {type(fil).__name__}; expected one of "
            f"{[t.__name__ for t in _OPT_CLASS_BY_FIL_TYPE]}"
        )
    return cls, fil


def make_under_topology_optimizer(fil, *, with_crit_sets, dims_to_restore_elz=None,
                                  n_threads=1, u_strategy=None):
    """Construct the C++ TopologyOptimizer for any filtration cell type."""
    cls, under_fil = _opt_class_for_filtration(fil)
    dims = list(dims_to_restore_elz) if dims_to_restore_elz else []
    if u_strategy is None:
        u_strategy = _oineus.UStrategy.Auto
    return cls(under_fil, with_crit_sets=with_crit_sets,
               dims_to_restore_elz=dims, n_threads=n_threads,
               u_strategy=u_strategy)


class TopologyOptimizer:
    """Thin Python wrapper that picks the right C++ TopologyOptimizer
    instantiation for the filtration's cell type and forwards calls
    to it. The under_fil (the C++ filtration) is exposed via
    `self.under_fil` for callers that need it directly.

    The optimizer is built for one autograd backward; the reduction
    recipe is fixed at construction time::

        with_crit_sets=False  -> R only (parallel + clearing).
        with_crit_sets=True   -> R + V + restore_ELZ in the given dims.
                                  U is recovered on demand via
                                  ensure_has_u_hom / ensure_has_u_coh,
                                  unless u_strategy=LegacyInBand, in
                                  which case U is built in-band
                                  (serial, clearing off).
    """

    def __init__(self, fil, *, with_crit_sets: bool = True,
                 dims_to_restore_elz: Optional[List[int]] = None,
                 n_threads: int = 1,
                 u_strategy=None):
        cls, under_fil = _opt_class_for_filtration(fil)
        dims = list(dims_to_restore_elz) if dims_to_restore_elz else []
        if u_strategy is None:
            u_strategy = _oineus.UStrategy.Auto
        self.under_opt = cls(under_fil, with_crit_sets=with_crit_sets,
                             dims_to_restore_elz=dims, n_threads=n_threads,
                             u_strategy=u_strategy)
        self.under_fil = under_fil

    # diagram + simplification

    def compute_diagram(self, include_inf_points: bool):
        return self.under_opt.compute_diagram(include_inf_points)

    def simplify(self, epsilon: float, strategy, dim: int):
        return self.under_opt.simplify(epsilon, strategy, dim)

    def get_nth_persistence(self, dim: int, n: int):
        return self.under_opt.get_nth_persistence(dim=dim, n=n)

    def match(self, template_dgm, dim: int, wasserstein_q: float = 1.0,
              wasserstein_delta: float = 0.01,
              return_wasserstein_distance: bool = False,
              dualize=None):
        if dualize is None:
            dualize = default_dualize_for_filtration(self.under_fil)
        if dualize:
            self.ensure_coh_reduced()
        else:
            self.ensure_hom_reduced()
        return self.under_opt.match(
            template_dgm=template_dgm,
            dim=dim,
            q=wasserstein_q,
            wasserstein_delta=wasserstein_delta,
            return_wasserstein_distance=return_wasserstein_distance,
            dualize=dualize,
        )

    # decomposition handles

    @property
    def homology_decomposition(self):
        return self.under_opt.homology_decomposition

    @property
    def cohomology_decomposition(self):
        return self.under_opt.cohomology_decomposition

    def homology_decomposition_ref(self):
        return self.under_opt.homology_decomposition_ref()

    def cohomology_decomposition_ref(self):
        return self.under_opt.cohomology_decomposition_ref()

    # reduction control

    def ensure_hom_built(self):
        return self.under_opt.ensure_hom_built()

    def ensure_coh_built(self):
        return self.under_opt.ensure_coh_built()

    @property
    def is_hom_built(self):
        return self.under_opt.is_hom_built

    @property
    def is_coh_built(self):
        return self.under_opt.is_coh_built

    def ensure_hom_reduced(self):
        return self.under_opt.ensure_hom_reduced()

    def ensure_coh_reduced(self):
        return self.under_opt.ensure_coh_reduced()

    def ensure_has_u_hom(self, dim: int, rows_fil, bounds):
        return self.under_opt.ensure_has_u_hom(dim, rows_fil, bounds)

    def ensure_has_u_coh(self, dim: int, rows_fil, bounds):
        return self.under_opt.ensure_has_u_coh(dim, rows_fil, bounds)

    def reduce_all(self):
        return self.under_opt.reduce_all()

    def update(self, new_values, n_threads: int = 1):
        return self.under_opt.update(new_values, n_threads)

    # crit-sets primitives

    def crit_sets_apply(self, indices, values, strategy):
        return self.under_opt.crit_sets_apply(indices, values, strategy)

    def singleton(self, index: int, value: float):
        return self.under_opt.singleton(index, value)

    def singletons(self, indices, values):
        return self.under_opt.singletons(indices, values)

    def combine_loss(self, critical_sets, strategy):
        return self.under_opt.combine_loss(critical_sets, strategy)

    def increase_death(self, negative_simplex_idx: int):
        return self.under_opt.increase_death(negative_simplex_idx)

    def decrease_death(self, negative_simplex_idx: int):
        return self.under_opt.decrease_death(negative_simplex_idx)

    def increase_birth(self, positive_simplex_idx: int):
        return self.under_opt.increase_birth(positive_simplex_idx)

    def decrease_birth(self, positive_simplex_idx: int):
        return self.under_opt.decrease_birth(positive_simplex_idx)
