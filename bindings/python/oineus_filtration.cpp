#include <functional>
#include "oineus_persistence_bindings.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/unordered_map.h>


nb::ndarray<size_t, nb::numpy> extract_simplices_as_numpy(const oin::Filtration<oin::Simplex<oin_int>, oin_real>& fil, dim_type simplex_dim)
{
    using VertexIndex = size_t;
    const dim_type simplex_size = simplex_dim + 1;
    const VertexIndex n_simplices = fil.size_in_dimension(simplex_dim);

    auto* simplices = new VertexIndex[simplex_size * n_simplices];

    for(auto simplex_idx = fil.dim_first(simplex_dim); simplex_idx <= fil.dim_last(simplex_dim); ++simplex_idx) {
        size_t array_idx = simplex_idx - fil.dim_first(simplex_dim);
        assert(fil.get_cell(simplex_idx).get_vertices().size() == simplex_size);
        for(size_t v_idx = 0; v_idx < simplex_size; ++v_idx) {
            simplices[simplex_size * array_idx + v_idx] = fil.get_cell(simplex_idx).get_vertices()[v_idx];
        }
    }

    nb::capsule free_when_done(simplices, [](void* p) noexcept {
     auto* pp = reinterpret_cast<VertexIndex*>(p);
     delete[] pp;
   });

    return nb::ndarray<VertexIndex, nb::numpy>(simplices, {n_simplices, simplex_size}, free_when_done);
}

void init_oineus_filtration(nb::module_& m)
{
    using Simplex = oin::Simplex<oin_int>;
    using Filtration = oin::Filtration<Simplex, oin_real>;

    using FiltrationStateTuple = std::tuple<decltype(Filtration::negate_),
                                            decltype(Filtration::cells_),
                                            decltype(Filtration::is_subfiltration_),
                                            decltype(Filtration::uid_to_sorted_id),
                                            decltype(Filtration::id_to_sorted_id_),
                                            decltype(Filtration::sorted_id_to_id_),
                                            decltype(Filtration::dim_first_),
                                            decltype(Filtration::dim_last_)
                                           >;


    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdFiltration = oin::Filtration<ProdSimplex, oin_real>;

    using ProdFiltrationStateTuple = std::tuple<decltype(ProdFiltration::negate_),
                                                decltype(ProdFiltration::cells_),
                                                decltype(ProdFiltration::is_subfiltration_),
                                                decltype(ProdFiltration::uid_to_sorted_id),
                                                decltype(ProdFiltration::id_to_sorted_id_),
                                                decltype(ProdFiltration::sorted_id_to_id_),
                                                decltype(ProdFiltration::dim_first_),
                                                decltype(ProdFiltration::dim_last_)
                                               >;

    using oin::VREdge;

    const std::string filtration_class_name = "Filtration";
    const std::string prod_filtration_class_name = "ProdFiltration";


    nb::class_<Filtration>(m, filtration_class_name.c_str())
            .def(nb::init<Filtration::CellVector, bool, int>(),
                    nb::arg("cells"),
                    nb::arg("negate") = false,
                    nb::arg("n_threads") = 1
                    )
            // this ctor accepts the output of Diode directly, list of (vertices, value)
            .def("__init__", [](Filtration* pfil, const std::vector<std::tuple<std::vector<unsigned>, oin_real>>& diode_simplices, int n_threads) {
                std::vector<Filtration::Cell> oin_simplices;
                oin_simplices.reserve(diode_simplices.size());
                for(const auto& [vs_, val] : diode_simplices) {
                    Simplex::IdxVector vs;
                    vs.reserve(vs_.size());
                    for (unsigned v : vs_) { vs.push_back(v); }
                    oin_simplices.emplace_back(Simplex(vs), val);
                }
                // Negation must stay false here; n_threads is the third ctor argument.
                new (pfil) Filtration(std::move(oin_simplices), false, n_threads);
            }, nb::arg("vertices_values"), nb::arg("n_threads") = 1)
            .def("__len__", &Filtration::size)
            .def("__iter__", [](Filtration& fil) { return nb::make_iterator(nb::type<Filtration>(), "simplex_iterator", fil.begin(), fil.end()); }, nb::keep_alive<0, 1>())
            .def("__getitem__", [](Filtration& fil, int i) { if (i < 0) i = fil.size() + i; return fil.get_cell(i);})
            .def_prop_ro("negate", &Filtration::negate)
            .def("max_dim", &Filtration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &Filtration::cells_copy, "copy of all cells in filtration order")
            .def("simplices", &Filtration::cells_copy, "copy of all simplices (cells) in filtration order")
            .def("size", &Filtration::size, "number of cells in filtration")
            .def("size_in_dimension", &Filtration::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim")
            .def("n_vertices", &Filtration::n_vertices)
            .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, nb::arg("sorted_id"))
            .def("id_by_sorted_id", &Filtration::get_id_by_sorted_id, nb::arg("sorted_id"))
            .def("sorted_id_by_id", &Filtration::get_sorted_id, nb::arg("id"))
            .def("cell", &Filtration::get_cell, nb::arg("i"))
            .def("simplex", &Filtration::get_cell, nb::arg("i"))
            .def_prop_ro("dim_first", &Filtration::dims_first)
            .def_prop_ro("dim_last", &Filtration::dims_last)
            .def("sorting_permutation", &Filtration::get_sorting_permutation)
            .def("inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .def("value_by_uid", &Filtration::value_by_uid, nb::arg("uid"))
            .def("sorted_id_by_uid", &Filtration::get_sorted_id_by_uid, nb::arg("uid"))
            .def("cell_by_uid", &Filtration::get_cell_by_uid, nb::arg("uid"))
            .def("boundary_matrix", &Filtration::boundary_matrix, nb::arg("n_threads")=1)
            .def("boundary_matrix_in_dimension", &Filtration::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads")=1)
            .def("coboundary_matrix", &Filtration::coboundary_matrix, nb::arg("n_threads")=1)
            .def("boundary_matrix_rel", &Filtration::boundary_matrix_rel)
            .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            .def("set_values", &Filtration::set_values, nb::arg("new_values"), nb::arg("n_threads")=1)
            .def("subfiltration", [](Filtration& self, const
                std::function<bool(const Simplex&)>& py_pred) -> Filtration {
                auto pred = [&py_pred](const Filtration::Cell& c) -> bool { return py_pred(c.cell_);
                };
                // auto pred = [](const Filtration::Cell& cell) -> bool { return cell.dim() == 1; };
                auto result = self.subfiltration(pred);
                return result;
             }, nb::arg("predicate"), nb::rv_policy::copy)
             // .def("subfiltration", [](Filtration& self, const
                // std::function<bool(const Filtration::Cell&)>& py_pred) {
                // Filtration result = self.subfiltration(py_pred);
                // return result;
             // }, nb::arg("predicate"))
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("get_vertices", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 0); })
            .def("get_edges", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 1); })
            .def("get_triangles", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 2); })
            .def("get_tetrahedra", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 3); })
            .def("get_simplices_as_arr", [](Filtration& self, dim_type simplex_dim) -> nb::ndarray<size_t, nb::numpy>
                { return extract_simplices_as_numpy(self, simplex_dim); }, nb::arg("simplex_dim"))
            .def("__repr__", [](const Filtration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            })
            .def("__getstate__", [](const Filtration& fil) -> FiltrationStateTuple {
                  return std::make_tuple(fil.negate_, fil.cells_, fil.is_subfiltration_,
                      fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_);
                })
            .def("__setstate__", [](Filtration& fil, const FiltrationStateTuple& t) {
                new (&fil) Filtration();
                fil.negate_ = std::get<0>(t);
                fil.cells_ = std::get<1>(t);
                fil.is_subfiltration_ = std::get<2>(t);
                fil.uid_to_sorted_id = std::get<3>(t);
                fil.id_to_sorted_id_ = std::get<4>(t);
                fil.sorted_id_to_id_ = std::get<5>(t);
                fil.dim_first_ = std::get<6>(t);
                fil.dim_last_ = std::get<7>(t);
            })
    ;

    nb::class_<ProdFiltration>(m, prod_filtration_class_name.c_str())
            .def(nb::init<ProdFiltration::CellVector, bool, int>(),
                    nb::arg("cells"),
                    nb::arg("negate") = false,
                    nb::arg("n_threads") = 1
                    )
            .def("__len__", &ProdFiltration::size)
            .def("__iter__", [](ProdFiltration& fil) { return nb::make_iterator(nb::type<ProdFiltration>(), "simplex_simplex_iterator", fil.begin(), fil.end()); }, nb::keep_alive<0,
            1>())
            .def("__getitem__", [](ProdFiltration& fil, size_t i) { return fil.get_cell(i);})
            .def_prop_ro("negate", &ProdFiltration::negate)
            .def("max_dim", &ProdFiltration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &ProdFiltration::cells_copy, "copy of all cells in filtration order")
            .def("size", &ProdFiltration::size, "number of cells in filtration")
            .def_prop_ro("dim_first", &ProdFiltration::dims_first)
            .def_prop_ro("dim_last", &ProdFiltration::dims_last)
            .def("size_in_dimension", &ProdFiltration::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim")
            .def("cell_value_by_sorted_id", &ProdFiltration::value_by_sorted_id, nb::arg("sorted_id"))
            .def("get_id_by_sorted_id", &ProdFiltration::get_id_by_sorted_id, nb::arg("sorted_id"))
            .def("get_sorted_id_by_id", &ProdFiltration::get_sorted_id, nb::arg("id"))
            .def("get_cell", &ProdFiltration::get_cell, nb::arg("i"))
            .def("get_sorting_permutation", &ProdFiltration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &ProdFiltration::get_inv_sorting_permutation)
            .def("boundary_matrix", &ProdFiltration::boundary_matrix, nb::arg("n_threads"))
            .def("boundary_matrix_in_dimension", &ProdFiltration::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads"))
            .def("coboundary_matrix", &ProdFiltration::coboundary_matrix, nb::arg("n_threads"))
            .def("reset_ids_to_sorted_ids", &ProdFiltration::reset_ids_to_sorted_ids)
            .def("set_values", &ProdFiltration::set_values, nb::arg("new_values"), nb::arg("n_threads")=1)
            .def("__repr__", [](const ProdFiltration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const ProdFiltration& fil) -> ProdFiltrationStateTuple {
                  return std::make_tuple(fil.negate_, fil.cells_, fil.is_subfiltration_,
                      fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_);
                })
            .def("__setstate__", [](ProdFiltration& fil, const ProdFiltrationStateTuple& t) {
                new (&fil) ProdFiltration();
                fil.negate_ = std::get<0>(t);
                fil.cells_ = std::get<1>(t);
                fil.is_subfiltration_ = std::get<2>(t);
                fil.uid_to_sorted_id = std::get<3>(t);
                fil.id_to_sorted_id_ = std::get<4>(t);
                fil.sorted_id_to_id_ = std::get<5>(t);
                fil.dim_first_ = std::get<6>(t);
                fil.dim_last_ = std::get<7>(t);
            })
    ;

    // ============ CubeFiltration bindings ============
    #define BIND_CUBE_FILTRATION(DIM) \
        using CubeFiltration_##DIM##D = oin::Filtration<oin::Cube<oin_int, DIM>, oin_real>; \
        using CubeFiltration_##DIM##DStateTuple = std::tuple<decltype(CubeFiltration_##DIM##D::negate_), \
                                                decltype(CubeFiltration_##DIM##D::cells_), \
                                                decltype(CubeFiltration_##DIM##D::is_subfiltration_), \
                                                decltype(CubeFiltration_##DIM##D::uid_to_sorted_id), \
                                                decltype(CubeFiltration_##DIM##D::id_to_sorted_id_), \
                                                decltype(CubeFiltration_##DIM##D::sorted_id_to_id_), \
                                                decltype(CubeFiltration_##DIM##D::dim_first_), \
                                                decltype(CubeFiltration_##DIM##D::dim_last_) \
                                               >; \
        nb::class_<CubeFiltration_##DIM##D>(m, "CubeFiltration_" #DIM "D") \
            .def(nb::init<CubeFiltration_##DIM##D::CellVector, bool, int>(), \
                    nb::arg("cells"), \
                    nb::arg("negate") = false, \
                    nb::arg("n_threads") = 1 \
                    ) \
            .def("__len__", &CubeFiltration_##DIM##D::size) \
            .def("__iter__", [](CubeFiltration_##DIM##D& fil) { \
                    return nb::make_iterator(nb::type<CubeFiltration_##DIM##D>(), "cube_iterator", fil.begin(), fil.end()); \
                }, nb::keep_alive<0, 1>()) \
            .def("__getitem__", [](CubeFiltration_##DIM##D& fil, int i) { \
                    if (i < 0) i = fil.size() + i; \
                    return fil.get_cell(i); \
                }) \
            .def_prop_ro("negate", &CubeFiltration_##DIM##D::negate) \
            .def("max_dim", &CubeFiltration_##DIM##D::max_dim, "maximal dimension of a cell in filtration") \
            .def("cells", &CubeFiltration_##DIM##D::cells_copy, "copy of all cells in filtration order") \
            .def("cubes", &CubeFiltration_##DIM##D::cells_copy, "copy of all cells in filtration order") \
            .def("size", &CubeFiltration_##DIM##D::size, "number of cells in filtration") \
            .def("size_in_dimension", &CubeFiltration_##DIM##D::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim") \
            .def("n_vertices", &CubeFiltration_##DIM##D::n_vertices) \
            .def("cube_value_by_sorted_id", &CubeFiltration_##DIM##D::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("id_by_sorted_id", &CubeFiltration_##DIM##D::get_id_by_sorted_id, nb::arg("sorted_id")) \
            .def("sorted_id_by_id", &CubeFiltration_##DIM##D::get_sorted_id, nb::arg("id")) \
            .def("cell", &CubeFiltration_##DIM##D::get_cell, nb::arg("i")) \
            .def("cube", &CubeFiltration_##DIM##D::get_cell, nb::arg("i")) \
            .def_prop_ro("dim_first", &CubeFiltration_##DIM##D::dims_first) \
            .def_prop_ro("dim_last", &CubeFiltration_##DIM##D::dims_last) \
            .def("sorting_permutation", &CubeFiltration_##DIM##D::get_sorting_permutation) \
            .def("inv_sorting_permutation", &CubeFiltration_##DIM##D::get_inv_sorting_permutation) \
            .def("value_by_uid", &CubeFiltration_##DIM##D::value_by_uid, nb::arg("uid")) \
            .def("sorted_id_by_uid", &CubeFiltration_##DIM##D::get_sorted_id_by_uid, nb::arg("uid")) \
            .def("cell_by_uid", &CubeFiltration_##DIM##D::get_cell_by_uid, nb::arg("uid")) \
            .def("boundary_matrix", &CubeFiltration_##DIM##D::boundary_matrix, nb::arg("n_threads")=1) \
            .def("boundary_matrix_in_dimension", &CubeFiltration_##DIM##D::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads")=1) \
            .def("coboundary_matrix", &CubeFiltration_##DIM##D::coboundary_matrix, nb::arg("n_threads")=1) \
            .def("boundary_matrix_rel", &CubeFiltration_##DIM##D::boundary_matrix_rel) \
            .def("reset_ids_to_sorted_ids", &CubeFiltration_##DIM##D::reset_ids_to_sorted_ids) \
            .def("set_values", &CubeFiltration_##DIM##D::set_values, nb::arg("new_values"), nb::arg("n_threads")=1) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def("__repr__", [](const CubeFiltration_##DIM##D& fil) { \
                    std::stringstream ss; \
                    ss << fil; \
                    return ss.str(); \
                }) \
            .def("__getstate__", [](const CubeFiltration_##DIM##D& fil) -> CubeFiltration_##DIM##DStateTuple { \
                      return std::make_tuple(fil.negate_, fil.cells_, fil.is_subfiltration_, \
                          fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_); \
                    }) \
            .def("__setstate__", [](CubeFiltration_##DIM##D& fil, const CubeFiltration_##DIM##DStateTuple& t) { \
                new (&fil) CubeFiltration_##DIM##D(); \
                fil.negate_ = std::get<0>(t); \
                fil.cells_ = std::get<1>(t); \
                fil.is_subfiltration_ = std::get<2>(t); \
                fil.uid_to_sorted_id = std::get<3>(t); \
                fil.id_to_sorted_id_ = std::get<4>(t); \
                fil.sorted_id_to_id_ = std::get<5>(t); \
                fil.dim_first_ = std::get<6>(t); \
                fil.dim_last_ = std::get<7>(t); \
            }) \


    BIND_CUBE_FILTRATION(1);
    BIND_CUBE_FILTRATION(2);
    BIND_CUBE_FILTRATION(3);

    #undef BIND_CUBE_FILTRATION

    m.def("_mapping_cylinder", &oin::build_mapping_cylinder<Simplex, oin_real>, nb::arg("fil_domain"), nb::arg("fil_codomain"), nb::arg("v_domain"), nb::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("_mapping_cylinder_with_indices", &oin::build_mapping_cylinder_with_indices<Simplex, oin_real>, nb::arg("fil_domain"), nb::arg("fil_codomain"), nb::arg("v_domain"), nb::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("_multiply_filtration", &oin::multiply_filtration<Simplex, oin_real>, nb::arg("fil"), nb::arg("sigma"), "return a filtration with each simplex in fil multiplied by simplex sigma");

    m.def("_min_filtration", &oin::min_filtration<Simplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a filtration where each simplex has minimal value from fil_1, fil_2");
    m.def("_min_filtration", &oin::min_filtration<ProdSimplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a filtration where each cell has minimal value from fil_1, fil_2");

    // helper for differentiable filtration
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<Simplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<ProdSimplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");

}
