from .. import _oineus
from .diff_filtration import DiffFiltration


_OPT_CLASS_BY_FIL_TYPE = {
    _oineus.Filtration:        _oineus.TopologyOptimizer,
    _oineus.ProdFiltration:    _oineus.TopologyOptimizerProd,
    _oineus.CubeFiltration_1D: _oineus.TopologyOptimizerCube_1D,
    _oineus.CubeFiltration_2D: _oineus.TopologyOptimizerCube_2D,
    _oineus.CubeFiltration_3D: _oineus.TopologyOptimizerCube_3D,
}


class TopologyOptimizer:
    """
    A wrapper around C++ topology optimizer.
    The C++ class is templated by the Cell type; this wrapper picks the
    correct instantiation from the type of the underlying filtration.
    """
    def __init__(self, fil):
        if isinstance(fil, DiffFiltration):
            fil = fil.under_fil

        cls = _OPT_CLASS_BY_FIL_TYPE.get(type(fil))
        if cls is None:
            raise RuntimeError(
                f"unknown filtration type {type(fil).__name__} in oineus.diff.TopologyOptimizer constructor"
            )
        self.under_opt = cls(fil)

    def compute_diagram(self, include_inf_points: bool):
        return self.under_opt.compute_diagram(include_inf_points)

    def simplify(self, epsilon: float, strategy, dim: int):
        return self.under_opt.simplify(epsilon, strategy, dim)

    def get_nth_persistence(self, dim: int, n: int):
        return self.under_opt.get_nth_persistence(dim=dim, n=n)

    def match(self, template_dgm, dim: int, wasserstein_q: float = 1.0, return_wasserstein_distance: bool = False):
        return self.under_opt.match(
            template_dgm=template_dgm,
            dim=dim,
            wasserstein_q=wasserstein_q,
            return_wasserstein_distance=return_wasserstein_distance,
        )

    @property
    def homology_decomposition(self):
        return self.under_opt.homology_decomposition

    @property
    def cohomology_decomposition(self):
        return self.under_opt.cohomology_decomposition

    def singleton(self, index: int, value: float):
        return self.under_opt.singleton(index, value)

    def singletons(self, indices, values):
        return self.under_opt.singletons(indices, values)

    def reduce_all(self):
        return self.under_opt.reduce_all()

    def update(self):
        return self.under_opt.update()

    def increase_death(self, negative_simplex_idx: int):
        return self.under_opt.increase_death(negative_simplex_idx)

    def decrease_death(self, negative_simplex_idx: int):
        return self.under_opt.decrease_death(negative_simplex_idx)

    def increase_birth(self, positive_simplex_idx: int):
        return self.under_opt.increase_birth(positive_simplex_idx)

    def decrease_birth(self, positive_simplex_idx: int):
        return self.under_opt.decrease_birth(positive_simplex_idx)

    def combine_loss(self, critical_sets, strategy):
        return self.under_opt.combine_loss(critical_sets, strategy)
