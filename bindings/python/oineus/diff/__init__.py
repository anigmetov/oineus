import functools as ft
from typing import Optional, Dict, Any, Union, Tuple
from icecream import ic

import numpy as np
import eagerpy as epy

from .. import _oineus
from .. import vr_filtration as non_diff_vr_filtration

from .diff_filtration import DiffFiltration

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .cech_delaunay import triangle_meb, tetrahedron_meb, cech_delaunay_filtration
    from .persistence_diagram import PersistenceDiagrams, persistence_diagram
    from .sliced_wasserstein import sliced_wasserstein_distance, sliced_wasserstein_distance_diag_corrected
    from .wasserstein import wasserstein_cost

# to copy docstring and name from the wrapped _oineus (C++) methods


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
        return self.under_opt.match(template_dgm=template_dgm, dim=dim, wasserstein_q=wasserstein_q,
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
    np_data = data.float64().numpy()
    fil, cv = _oineus.get_freudenthal_filtration_and_crit_vertices(np_data, negate, wrap, max_dim, n_threads)
    cv = np.array(cv, dtype=np.int64)
    values = data.flatten()[cv].raw
    return DiffFiltration(fil, values)


def vr_filtration(data, from_pwdists: bool=False, max_dim: int=-1, max_diameter: float=-1.0, eps=1e-6, n_threads=8) -> DiffFiltration:

    data = epy.astensor(data)
    data_np = data.float64().numpy()
    assert(data.ndim == 2)

    fil, edges = non_diff_vr_filtration(data=data_np, from_pwdists=from_pwdists, with_critical_edges=True,
                                        max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)

    if not from_pwdists:
        sqdists = epy.sum((data[edges[:, 0].flatten()] - data[edges[:, 1].flatten()]) ** 2, axis=1) + eps
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
        v_domain = v_domain.combinatorial_cell

    if isinstance(v_codomain, _oineus.Simplex):
        v_codomain = v_codomain.combinatorial_cell

    under_fil_dom = fil_domain.under_fil
    under_fil_cod = fil_codomain.under_fil

    under_cyl_fil, cyl_val_inds = _oineus._mapping_cylinder_with_indices(under_fil_dom, under_fil_cod, v_domain, v_codomain)

    cyl_val_inds = epy.astensor(np.array(cyl_val_inds, dtype=np.int64))

    concat_vals = epy.concatenate((epy.astensor(fil_domain.values), epy.astensor(fil_codomain.values)))

    assert(concat_vals.ndim == 1 and concat_vals.shape[0] == fil_domain.size() + fil_codomain.size())

    cyl_values = concat_vals[cyl_val_inds].raw

    return DiffFiltration(under_cyl_fil, cyl_values)
