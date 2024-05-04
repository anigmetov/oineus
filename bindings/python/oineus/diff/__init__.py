import functools as ft

import numpy as np
import eagerpy as epy

from .. import _oineus

class DiffFiltration:
    def __init__(self, fil, values):
        self.under_fil = fil
        self.values = values

    def __len__(self):
        return len(self.under_fil)

    def max_dim(self):
        return self.under_fil.max_dim()

    def size(self):
        return self.under_fil.size()

    def size_in_dimension(self, dim):
        return self.under_fil.size(dim)

    def n_vertices(self):
        return self.under_fil.n_vertices()

    def cells(self):
        return self.under_fil.cells()

    def get_id_by_sorted_id(self, sorted_id):
        return self.under_fil.get_id_by_sorted_id(sorted_id)

    def get_sorted_id_by_id(self, id):
        return self.under_fil.get_sorted_id_by_id(id)

    def get_cell(self, sorted_idx):
        return self.under_fil.get_cell(sorted_idx)

    def get_simplex(self, sorted_idx):
        return self.under_fil.get_simplex(sorted_idx)

    def get_sorting_permutation(self):
        return self.under_fil.get_sorting_permutation()

    def get_inv_sorting_permutation(self):
        return self.under_fil.get_inv_sorting_permutation()

    def cell_by_uid(self, uid):
        return self.under_fil.cell_by_uid(uid)

    def boundary_matrix(self, uid):
        return self.under_fil.boundary_matrix(uid)

            # .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, py::arg("sorted_id"))
            # .def("simplex_value_by_vertices", &Filtration::value_by_vertices, py::arg("vertices"))
            # .def("get_sorted_id_by_vertices", &Filtration::get_sorted_id_by_vertices, py::arg("vertices"))
            # .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            # .def("__repr__", [](const Filtration& fil) {



def min_filtration(fil_1: DiffFiltration, fil_2: DiffFiltration) -> DiffFiltration:
    fil_1_under = fil_1.under_fil
    fil_2_under = fil_2.under_fil

    min_fil_under, inds_1, inds_2 = _oineus.min_filtration_with_indices(fil_1_under, fil_2_under)

    inds_1 = np.array(inds_1)
    inds_2 = np.array(inds_2)

    vals_1 = epy.astensor(fil_1.values)[inds_1]
    vals_2 = epy.astensor(fil_2.values)[inds_2]

    min_fil_values = epy.min(epy.stack((vals_1, vals_2)), axis=0).raw

    return DiffFiltration(min_fil_under, min_fil_values)


def lower_star_freudenthal(data, negate, wrap, max_dim, n_threads):
    data = epy.astensor(data)
    dim_part = data.ndim
    np_data = data.float64().numpy()
    type_part = "double"
    func = getattr(_oineus, f"get_fr_filtration_and_critical_vertices_{type_part}_{dim_part}")
    fil, cv = func(np_data, negate, wrap, max_dim, n_threads)
    cv = np.array(cv, dtype=np.int64)
    values = data[cv].raw
    return DiffFiltration(fil, values)


def vietoris_rips_pts(pts, max_dim, max_radius, eps=1e-6, n_threads=1) -> DiffFiltration:
    pts = epy.astensor(pts)

    pts_np = pts.float64().numpy()

    type_part = "double"
    # if pts.dtype.name == "float32":
    #     type_part = "float"
    # elif pts.dtype.name == "float64":
    #     type_part = "double"
    # else:
    #     raise RuntimeError(f"Unknown datatype: {pts.dtype}")
    assert(pts.ndim == 2)
    dim_part = pts.shape[1]

    func = getattr(_oineus, f"get_vr_filtration_and_critical_edges_{type_part}_{dim_part}")

    fil, edges = func(pts_np, max_dim, max_radius, n_threads)
    edges_s = np.array([e.x for e in edges])
    edges_t = np.array([e.y for e in edges])

    sqdists = epy.sum((pts[edges_s] - pts[edges_t]) ** 2 + eps, axis=1)
    dists = epy.sqrt(sqdists).raw

    return DiffFiltration(fil, dists)


def vietoris_rips_pwdists(pwdists, max_dim, max_radius, eps=1e-6, n_threads=1):

    pwdists = epy.astensor(pwdists)

    assert(pwdists.ndim == 2 and pwdists.shape[0] == pwdists.shape[1])

    if pwdists.dtype.name == "float32":
        type_part = "float"
    elif pwdists.dtype.name == "float64":
        type_part = "double"
    else:
        raise RuntimeError(f"Unknown datatype: {pts.dtype}")

    func = getattr(_oineus, f"get_vr_filtration_and_critical_edges_from_pwdists_{type_part}")
    fil, edges = func(pwdists, max_dim, max_radius, n_threads)

    edges = epy.astensor(np.array([[e.x, e.y] for e in edges]), dtype=np.int32)

    dists = pwdists[edges[:, 0], edges[:, 1]].raw

    return DiffFiltration(fil, dists)


# build_mapping_cylinder_with_indices(const Filtration<Cell, Real>& fil_domain, const Filtration<Cell, Real>& fil_codomain, const Simplex<typename Cell::Int>& v_domain, const Simplex<typename Cell::Int>& v_codomain)
# def mapping_cylinder_filtration(fil_domain, fil_codomain, v_domain, v_codomain) -> DiffFiltration:
#
#     if type(fil_domain) is DiffFiltration:
#         fil_domain = fil_domain.under_fil
#
#     if type(fil_codomain) is DiffFiltration:
#         fil_codomain = fil_domain.under_fil
#
#     under_fil, indices = oin.build_mapping_cylinder_with_indices(fil_domain, fil_codomain, v_domain, v_codomain)
#
#
