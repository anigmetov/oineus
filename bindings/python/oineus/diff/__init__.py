import functools as ft

import numpy as np
import eagerpy as epy

from .. import _oineus
from .. import vr_filtration as non_diff_vr_filtration

# to copy docstring and name from the wrapped _oineus (C++) methods


class DiffFiltration:
    def __init__(self, fil, values):
        self.under_fil = fil
        self.values = values

    def __len__(self):
        return len(self.under_fil)

    def __repr__(self):
        return f"DiffFil(under_fil={self.under_fil}, values={self.values})"

    def __iter__(self):
        return iter(self.under_fil)

    #@ft.wraps(_oineus.Filtration.max_dim)
    def max_dim(self):
        return self.under_fil.max_dim()

    #@ft.wraps(_oineus.Filtration.size)
    def size(self):
        return self.under_fil.size()

    #@ft.wraps(_oineus.Filtration.size_in_dimension)
    def size_in_dimension(self, dim):
        return self.under_fil.size(dim)

    #@ft.wraps(_oineus.Filtration.n_vertices)
    def n_vertices(self):
        return self.under_fil.n_vertices()

    #@ft.wraps(_oineus.Filtration.cells)
    def cells(self):
        return self.under_fil.cells()

    #@ft.wraps(_oineus.Filtration.get_id_by_sorted_id)
    def id_by_sorted_id(self, sorted_id):
        return self.under_fil.id_by_sorted_id(sorted_id)

    #@ft.wraps(_oineus.Filtration.get_sorted_id_by_id)
    def sorted_id_by_id(self, id):
        return self.under_fil.sorted_id_by_id(id)

    #@ft.wraps(_oineus.Filtration.get_cell)
    def cell(self, sorted_idx):
        return self.under_fil.cell(sorted_idx)

    #@ft.wraps(_oineus.Filtration.get_simplex)
    def simplex(self, sorted_idx):
        return self.under_fil.simplex(sorted_idx)

    #@ft.wraps(_oineus.Filtration.get_sorting_permutation)
    def sorting_permutation(self):
        return self.under_fil.sorting_permutation()

    #@ft.wraps(_oineus.Filtration.get_inv_sorting_permutation)
    def get_inv_sorting_permutation(self):
        return self.under_fil.inv_sorting_permutation()

    #@ft.wraps(_oineus.Filtration.cell_by_uid)
    def cell_by_uid(self, uid):
        return self.under_fil.cell_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.boundary_matrix)
    def boundary_matrix(self, uid):
        return self.under_fil.boundary_matrix(uid)

    #@ft.wraps(_oineus.Filtration.simplex_value_by_sorted_id)
    def simplex_value_by_sorted_id(self, sorted_id):
        return self.under_fil.simplex_value_by_sorted_id(sorted_id)

    #@ft.wraps(_oineus.Filtration.simplex_value_by_vertices)
    def simplex_value_by_uid(self, uid):
        return self.under_fil.simplex_value_by_uid(uid)

    def cell_value_by_uid(self, uid):
        return self.under_fil.cell_value_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.get_sorted_id_by_vertices)
    def sorted_id_by_uid(self, uid):
        return self.under_fil.sorted_id_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.reset_ids_to_sorted_ids)
    def reset_ids_to_sorted_ids(self):
        self.under_fil.reset_ids_to_sorted_ids()


class TopologyOptimizer:
    """
    A wrapper around C++ topology optimizer
    The C++ class is templated by the Cell type,
    the purpose of this class is to allow the user
    not to worry about the type of a filtration
    """
    def __init__(self, fil):
        if isinstance(fil, DiffFiltration):
            fil = fil.under_fil
        if type(fil) is _oineus.Filtration:
            self.under_opt = _oineus.TopologyOptimizer(fil)
        elif type(fil) is _oineus.ProdFiltration:
            self.under_opt = _oineus.TopologyOptimizerProd(fil)
        else:
            raise RuntimeError("unknown filtration type in oineus.diff.TopologyOptimizer constructor")

    #@ft.wraps(_oineus.TopologyOptimizer.compute_diagram)
    def compute_diagram(self, include_inf_points: bool):
        return self.under_opt.compute_diagram(include_inf_points)

    #@ft.wraps(_oineus.TopologyOptimizer.simplify)
    def simplify(self, epsilon: float, strategy, dim: int):
        return self.under_opt.simplify(epsilon, strategy, dim)

    #@ft.wraps(_oineus.TopologyOptimizer.get_nth_persistence)
    def get_nth_persistence(self, dim: int, n: int):
        return self.under_opt.get_nth_persistence(dim=dim, n=n)

    #@ft.wraps(_oineus.TopologyOptimizer.match)
    def match(self, template_dgm, dim: int, wasserstein_q: float=1.0, return_wasserstein_distance: bool=False):
        return self.under_opt.match(temlate_dgm=template_dgm, dim=dim, wasserstein_q=wasserstein_q,
                                    return_wasserstein_distance=return_wasserstein_distance)

    @property
    def homology_decomposition(self):
        return self.under_opt.homology_decomposition

    @property
    def cohomology_decomposition(self):
        return self.under_opt.cohomology_decomposition

    #@ft.wraps(_oineus.TopologyOptimizer.singleton)
    def singleton(self, index: int, value: float):
        return self.under_opt.singleton(index, value)

    #@ft.wraps(_oineus.TopologyOptimizer.singletons)
    def singletons(self, indices, values):
        return self.under_opt.singletons(indices, values)

    #@ft.wraps(_oineus.TopologyOptimizer.reduce_all)
    def reduce_all(self):
        return self.under_opt.reduce_all()

    #@ft.wraps(_oineus.TopologyOptimizer.update)
    def update(self):
        return self.under_opt.update()

    #@ft.wraps(_oineus.TopologyOptimizer.increase_death)
    def increase_death(self, negative_simplex_idx: int):
        return self.under_opt.increase_death(negative_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.decrease_death)
    def decrease_death(self, negative_simplex_idx: int):
        return self.under_opt.decrease_death(negative_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.increase_birth)
    def increase_birth(self, positive_simplex_idx: int):
        return self.under_opt.increase_birth(positive_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.decrease_birth)
    def decrease_birth(self, positive_simplex_idx: int):
        return self.under_opt.decrease_birth(positive_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.combine_loss)
    def combine_loss(self, critical_sets, strategy):
        return self.under_opt.combine_loss(critical_sets, strategy)


def min_filtration(fil_1: DiffFiltration, fil_2: DiffFiltration) -> DiffFiltration:
    fil_1_under = fil_1.under_fil
    fil_2_under = fil_2.under_fil

    min_fil_under, inds_1, inds_2 = _oineus._min_filtration_with_indices(fil_1_under, fil_2_under)

    inds_1 = np.array(inds_1)
    inds_2 = np.array(inds_2)

    vals_1 = epy.astensor(fil_1.values)[inds_1]
    vals_2 = epy.astensor(fil_2.values)[inds_2]

    min_fil_values = epy.min(epy.stack((vals_1, vals_2)), axis=0).raw

    return DiffFiltration(min_fil_under, min_fil_values)


def freudenthal_filtration(data, negate, wrap, max_dim, n_threads):
    data = epy.astensor(data)
    dim_part = data.ndim
    np_data = data.float64().numpy()
    type_part = "double"
    func = getattr(_oineus, f"get_fr_filtration_and_critical_vertices_{type_part}_{dim_part}")
    fil, cv = func(np_data, negate, wrap, max_dim, n_threads)
    cv = np.array(cv, dtype=np.int64)
    values = data[cv].raw
    return DiffFiltration(fil, values)


def vr_filtration(data, from_pwdists: bool=False, max_dim: int=-1, max_diameter: float=-1.0, eps=1e-6, n_threads=8) -> DiffFiltration:

    data = epy.astensor(data)
    data_np = data.float64().numpy()
    assert(data.ndim == 2)

    fil, edges = non_diff_vr_filtration(data=data_np, from_pwdists=from_pwdists, with_critical_edges=True,
                                        max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)

    if not from_pwdists:
        sqdists = epy.sum((data[edges[:, 0].flatten()] - data[edges[:, 1].flatten()]) ** 2 + eps, axis=1)
        diff_dists = epy.sqrt(sqdists).raw

        return DiffFiltration(fil, diff_dists)
    else:
        edges = epy.astensor(edges)
        diff_dists = data[edges[:, 0], edges[:, 1]].raw
        return DiffFiltration(fil, diff_dists)


def mapping_cylinder_filtration(fil_domain: DiffFiltration, fil_codomain: DiffFiltration, v_domain, v_codomain) -> DiffFiltration:
    assert(type(fil_domain) is DiffFiltration)
    assert(type(fil_codomain) is DiffFiltration)

    if isinstance(v_domain, _oineus.Simplex):
        v_domain = v_domain.combinatorial_cell()

    if isinstance(v_codomain, _oineus.Simplex):
        v_codomain = v_codomain.combinatorial_cell()

    under_fil_dom = fil_domain.under_fil
    under_fil_cod = fil_codomain.under_fil

    under_cyl_fil, cyl_val_inds = _oineus._mapping_cylinder_with_indices(under_fil_dom, under_fil_cod, v_domain, v_codomain)

    cyl_val_inds = epy.astensor(np.array(cyl_val_inds, dtype=np.int64))

    concat_vals = epy.concatenate((epy.astensor(fil_domain.values), epy.astensor(fil_codomain.values)))

    assert(concat_vals.ndim == 1 and concat_vals.shape[0] == fil_domain.size() + fil_codomain.size())

    cyl_values = concat_vals[cyl_val_inds].raw

    return DiffFiltration(under_cyl_fil, cyl_values)

