#include "oineus_persistence_bindings.h"

void init_oineus_filtration(py::module& m)
{
    using namespace pybind11::literals;

    using Simplex = oin::Simplex<oin_int>;
    using Filtration = oin::Filtration<Simplex, oin_real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdFiltration = oin::Filtration<ProdSimplex, oin_real>;

    using oin::VREdge;

    const std::string filtration_class_name = "Filtration";
    const std::string prod_filtration_class_name = "ProdFiltration";

    py::class_<Filtration>(m, filtration_class_name.c_str())
            .def(py::init<Filtration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate") = false,
                    py::arg("n_threads") = 1,
                    py::arg("sort_only_by_dimension") = false,
                    py::arg("set_ids") = true)
            .def("__len__", &Filtration::size)
            .def("__iter__", [](Filtration& fil) { return py::make_iterator(fil.begin(), fil.end()); }, py::keep_alive<0, 1>())
            .def("__getitem__", [](Filtration& fil, int i) { if (i < 0) i = fil.size() + i; return fil.get_cell(i);})
            .def_property_readonly("negate", &Filtration::negate)
            .def("max_dim", &Filtration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &Filtration::cells_copy, "copy of all cells in filtration order")
            .def("simplices", &Filtration::cells_copy, "copy of all simplices (cells) in filtration order")
            .def("size", &Filtration::size, "number of cells in filtration")
            .def("size_in_dimension", &Filtration::size_in_dimension, py::arg("dim"), "number of cells of dimension dim")
            .def("n_vertices", &Filtration::n_vertices)
            .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, py::arg("sorted_id"))
            .def("id_by_sorted_id", &Filtration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("sorted_id_by_id", &Filtration::get_sorted_id, py::arg("id"))
            .def("cell", &Filtration::get_cell, py::arg("i"))
            .def("simplex", &Filtration::get_cell, py::arg("i"))
            .def("dim_first", [](Filtration& fil, int d) { return fil.dim_first(d); }, py::arg("d"))
            .def("sorting_permutation", &Filtration::get_sorting_permutation)
            .def("inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .def("value_by_uid", &Filtration::value_by_uid, py::arg("uid"))
            .def("sorted_id_by_uid", &Filtration::get_sorted_id_by_uid, py::arg("uid"))
            .def("cell_by_uid", &Filtration::get_cell_by_uid, py::arg("uid"))
            .def("boundary_matrix", &Filtration::boundary_matrix, py::arg("n_threads")=1)
            .def("boundary_matrix_in_dimension", &Filtration::boundary_matrix_in_dimension, py::arg("dim"), py::arg("n_threads")=1)
            .def("coboundary_matrix", &Filtration::coboundary_matrix, py::arg("n_threads")=1)
            .def("boundary_matrix_rel", &Filtration::boundary_matrix_rel)
            .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            .def("set_values", &Filtration::set_values)
            .def("subfiltration", [](Filtration& self, const py::function& py_pred) {
                auto pred = [&py_pred](const typename Filtration::Cell& s) -> bool { return py_pred(s).template cast<bool>(); };
                Filtration result = self.subfiltration(pred);
                return result;
            }, py::arg("predicate"), py::return_value_policy::move)
            .def("__repr__", [](const Filtration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            });

    py::class_<ProdFiltration>(m, prod_filtration_class_name.c_str())
            .def(py::init<ProdFiltration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate") = false,
                    py::arg("n_threads") = 1,
                    py::arg("sort_only_by_dimension") = false,
                    py::arg("set_ids") = true)
            .def("__len__", &ProdFiltration::size)
            .def("__iter__", [](ProdFiltration& fil) { return py::make_iterator(fil.begin(), fil.end()); }, py::keep_alive<0, 1>())
            .def("__getitem__", [](ProdFiltration& fil, size_t i) { return fil.get_cell(i);})
            .def_property_readonly("negate", &ProdFiltration::negate)
            .def("max_dim", &ProdFiltration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &ProdFiltration::cells_copy, "copy of all cells in filtration order")
            .def("size", &ProdFiltration::size, "number of cells in filtration")
            .def("size_in_dimension", &ProdFiltration::size_in_dimension, py::arg("dim"), "number of cells of dimension dim")
            .def("cell_value_by_sorted_id", &ProdFiltration::value_by_sorted_id, py::arg("sorted_id"))
            .def("get_id_by_sorted_id", &ProdFiltration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("get_sorted_id_by_id", &ProdFiltration::get_sorted_id, py::arg("id"))
            .def("get_cell", &ProdFiltration::get_cell, py::arg("i"))
            .def("get_sorting_permutation", &ProdFiltration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &ProdFiltration::get_inv_sorting_permutation)
            .def("boundary_matrix", &ProdFiltration::boundary_matrix, py::arg("n_threads"))
            .def("boundary_matrix_in_dimension", &ProdFiltration::boundary_matrix_in_dimension, py::arg("dim"), py::arg("n_threads"))
            .def("coboundary_matrix", &ProdFiltration::coboundary_matrix, py::arg("n_threads"))
            .def("reset_ids_to_sorted_ids", &ProdFiltration::reset_ids_to_sorted_ids)
            .def("set_values", &ProdFiltration::set_values)
            .def("__repr__", [](const ProdFiltration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            });

    // Type aliases
    using CubeFiltration_1D = oin::Filtration<oin::Cube<oin_int, 1>, oin_real>;
    using CubeFiltration_2D = oin::Filtration<oin::Cube<oin_int, 2>, oin_real>;
    using CubeFiltration_3D = oin::Filtration<oin::Cube<oin_int, 3>, oin_real>;

    // ============ CubeFiltration bindings ============
    #define BIND_CUBE_FILTRATION(DIM) \
        py::class_<CubeFiltration_##DIM##D>(m, "CubeFiltration_" #DIM "D") \
            .def(py::init<CubeFiltration_##DIM##D::CellVector, bool, int, bool, bool>(), \
                    py::arg("cells"), \
                    py::arg("negate") = false, \
                    py::arg("n_threads") = 1, \
                    py::arg("sort_only_by_dimension") = false, \
                    py::arg("set_ids") = true) \
            .def("__len__", &CubeFiltration_##DIM##D::size) \
            .def("__iter__", [](CubeFiltration_##DIM##D& fil) { \
                    return py::make_iterator(fil.begin(), fil.end()); \
                }, py::keep_alive<0, 1>()) \
            .def("__getitem__", [](CubeFiltration_##DIM##D& fil, int i) { \
                    if (i < 0) i = fil.size() + i; \
                    return fil.get_cell(i); \
                }) \
            .def_property_readonly("negate", &CubeFiltration_##DIM##D::negate) \
            .def("max_dim", &CubeFiltration_##DIM##D::max_dim, "maximal dimension of a cell in filtration") \
            .def("cells", &CubeFiltration_##DIM##D::cells_copy, "copy of all cells in filtration order") \
            .def("cubes", &CubeFiltration_##DIM##D::cells_copy, "copy of all cells in filtration order") \
            .def("size", &CubeFiltration_##DIM##D::size, "number of cells in filtration") \
            .def("size_in_dimension", &CubeFiltration_##DIM##D::size_in_dimension, py::arg("dim"), "number of cells of dimension dim") \
            .def("n_vertices", &CubeFiltration_##DIM##D::n_vertices) \
            .def("cube_value_by_sorted_id", &CubeFiltration_##DIM##D::value_by_sorted_id, py::arg("sorted_id")) \
            .def("id_by_sorted_id", &CubeFiltration_##DIM##D::get_id_by_sorted_id, py::arg("sorted_id")) \
            .def("sorted_id_by_id", &CubeFiltration_##DIM##D::get_sorted_id, py::arg("id")) \
            .def("cell", &CubeFiltration_##DIM##D::get_cell, py::arg("i")) \
            .def("cube", &CubeFiltration_##DIM##D::get_cell, py::arg("i")) \
            .def("dim_first", [](CubeFiltration_##DIM##D& fil, int d) { return fil.dim_first(d); }, py::arg("d")) \
            .def("sorting_permutation", &CubeFiltration_##DIM##D::get_sorting_permutation) \
            .def("inv_sorting_permutation", &CubeFiltration_##DIM##D::get_inv_sorting_permutation) \
            .def("value_by_uid", &CubeFiltration_##DIM##D::value_by_uid, py::arg("uid")) \
            .def("sorted_id_by_uid", &CubeFiltration_##DIM##D::get_sorted_id_by_uid, py::arg("uid")) \
            .def("cell_by_uid", &CubeFiltration_##DIM##D::get_cell_by_uid, py::arg("uid")) \
            .def("boundary_matrix", &CubeFiltration_##DIM##D::boundary_matrix, py::arg("n_threads")=1) \
            .def("boundary_matrix_in_dimension", &CubeFiltration_##DIM##D::boundary_matrix_in_dimension, py::arg("dim"), py::arg("n_threads")=1) \
            .def("coboundary_matrix", &CubeFiltration_##DIM##D::coboundary_matrix, py::arg("n_threads")=1) \
            .def("boundary_matrix_rel", &CubeFiltration_##DIM##D::boundary_matrix_rel) \
            .def("reset_ids_to_sorted_ids", &CubeFiltration_##DIM##D::reset_ids_to_sorted_ids) \
            .def("set_values", &CubeFiltration_##DIM##D::set_values) \
            .def("subfiltration", [](CubeFiltration_##DIM##D& self, const py::function& py_pred) { \
                    auto pred = [&py_pred](const typename CubeFiltration_##DIM##D::Cell& s) -> bool { \
                        return py_pred(s).template cast<bool>(); \
                    }; \
                    CubeFiltration_##DIM##D result = self.subfiltration(pred); \
                    return result; \
                }, py::arg("predicate"), py::return_value_policy::move) \
            .def("__repr__", [](const CubeFiltration_##DIM##D& fil) { \
                    std::stringstream ss; \
                    ss << fil; \
                    return ss.str(); \
                })

    BIND_CUBE_FILTRATION(1);
    BIND_CUBE_FILTRATION(2);
    BIND_CUBE_FILTRATION(3);

    #undef BIND_CUBE_FILTRATION

    m.def("_mapping_cylinder", &oin::build_mapping_cylinder<Simplex, oin_real>, py::arg("fil_domain"), py::arg("fil_codomain"), py::arg("v_domain"), py::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("_mapping_cylinder_with_indices", &oin::build_mapping_cylinder_with_indices<Simplex, oin_real>, py::arg("fil_domain"), py::arg("fil_codomain"), py::arg("v_domain"), py::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("_multiply_filtration", &oin::multiply_filtration<Simplex, oin_real>, py::arg("fil"), py::arg("sigma"), "return a filtration with each simplex in fil multiplied by simplex sigma");

    m.def("_min_filtration", &oin::min_filtration<Simplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a filtration where each simplex has minimal value from fil_1, fil_2");
    m.def("_min_filtration", &oin::min_filtration<ProdSimplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a filtration where each cell has minimal value from fil_1, fil_2");

    // helper for differentiable filtration
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<Simplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<ProdSimplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");

}
