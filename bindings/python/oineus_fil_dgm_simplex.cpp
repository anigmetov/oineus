#include "oineus_persistence_bindings.h"

void init_oineus_fil_dgm_simplex(py::module& m)
{
    using namespace pybind11::literals;

    using DgmPoint = typename oin::Diagrams<oin_real>::Point;
    using DgmPtVec = typename oin::Diagrams<oin_real>::Dgm;
    using IndexDgmPtVec = typename oin::Diagrams<oin_real>::IndexDgm;
    using Diagram = PyOineusDiagrams<oin_real>;

    using Simplex = oin::Simplex<oin_int>;
    using SimplexValue = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>;
    using Filtration = oin::Filtration<Simplex, oin_real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexValue = oin::CellWithValue<ProdSimplex, oin_real>;
    using ProdFiltration = oin::Filtration<ProdSimplex, oin_real>;

    using oin::VREdge;

    using VRUDecomp = oin::VRUDecomposition<oin_int>;
    using KerImCokRedSimplex = oin::KerImCokReduced<Simplex, oin_real>;
    using KerImCokRedProdSimplex = oin::KerImCokReduced<ProdSimplex, oin_real>;

    std::string filtration_class_name = "Filtration";
    std::string prod_filtration_class_name = "ProdFiltration";
    std::string simplex_class_name = "Simplex";
    std::string prod_simplex_class_name = "ProdSimplex";

    std::string dgm_point_name = "DiagramPoint";
    std::string dgm_class_name = "Diagrams";

    std::string ker_im_cok_reduced_class_name = "KerImCokReduced";
    std::string ker_im_cok_reduced_prod_class_name = "KerImCokReducedProd";

    py::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<oin_real, oin_real>(), py::arg("birth"), py::arg("death"))
            .def_readwrite("birth", &DgmPoint::birth)
            .def_readwrite("death", &DgmPoint::death)
            .def_readwrite("birth_index", &DgmPoint::birth_index)
            .def_readwrite("death_index", &DgmPoint::death_index)
            .def("__getitem__", [](const DgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPoint& p) { return std::hash<DgmPoint>()(p); })
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<Diagram>(m, dgm_class_name.c_str())
            .def(py::init<dim_type>())
            .def("in_dimension", [](Diagram& self, dim_type dim, bool as_numpy) -> std::variant<pybind11::array_t<oin_real>, DgmPtVec> {
                      if (as_numpy)
                          return self.get_diagram_in_dimension_as_numpy(dim);
                      else
                          return self.get_diagram_in_dimension(dim);
                    }, "return persistence diagram in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    py::arg("dim"), py::arg("as_numpy") = true)
            .def("index_diagram_in_dimension", [](Diagram& self, dim_type dim, bool as_numpy, bool sorted) -> std::variant<pybind11::array_t<size_t>, IndexDgmPtVec> {
                      if (as_numpy)
                          return self.get_index_diagram_in_dimension_as_numpy(dim, sorted);
                      else
                          return self.get_index_diagram_in_dimension(dim, sorted);
                    }, "return persistence pairing (index diagram) in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    py::arg("dim"), py::arg("as_numpy") = true, py::arg("sorted") =  true)
            .def("__getitem__", &Diagram::get_diagram_in_dimension_as_numpy);

    py::class_<SimplexValue>(m, simplex_class_name.c_str())
            .def(py::init([](typename Simplex::IdxVector vs, oin_real value) -> SimplexValue {
                      return SimplexValue({vs}, value);
                    }),
                    py::arg("vertices"),
                    py::arg("value"))
            .def(py::init([](oin_int id, typename Simplex::IdxVector vs, oin_real value) -> SimplexValue { return SimplexValue({id, vs}, value); }), py::arg("id"), py::arg("vertices"), py::arg("value"))
            .def_property("id", &SimplexValue::get_id, &SimplexValue::set_id)
            .def_readwrite("sorted_id", &SimplexValue::sorted_id_)
            .def_property_readonly("vertices", [](const SimplexValue& sigma) { return sigma.cell_.vertices_; })
            .def_readwrite("value", &SimplexValue::value_)
            .def("dim", &SimplexValue::dim)
            .def("boundary", &SimplexValue::boundary)
            .def("join", [](const SimplexValue& sigma, oin_int new_vertex, oin_real value, oin_int new_id) {
                      return sigma.join(new_id, new_vertex, value);
                    },
                    py::arg("new_vertex"),
                    py::arg("value"),
                    py::arg("new_id") = SimplexValue::k_invalid_id)
            .def("__repr__", [](const SimplexValue& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    py::class_<ProdSimplexValue>(m, prod_simplex_class_name.c_str())
            .def(py::init([](const SimplexValue& sigma, const SimplexValue& tau, oin_real value) -> ProdSimplexValue {
                      return ProdSimplexValue(ProdSimplex(sigma.get_cell(), tau.get_cell()), value);
                    }),
                    py::arg("cell_1"),
                    py::arg("cell_2"),
                    py::arg("value"))
            .def(py::init([](const Simplex& sigma, const Simplex& tau, oin_real value) -> ProdSimplexValue {
                      return ProdSimplexValue(ProdSimplex(sigma, tau), value);
                    }),
                    py::arg("cell_1"),
                    py::arg("cell_2"),
                    py::arg("value"))
            .def_property("id", &ProdSimplexValue::get_id, &ProdSimplexValue::set_id)
            .def_readwrite("sorted_id", &ProdSimplexValue::sorted_id_)
            .def_property_readonly("cell_1", &ProdSimplexValue::get_factor_1)
            .def_property_readonly("cell_2", &ProdSimplexValue::get_factor_2)
            .def_property_readonly("uid", &ProdSimplexValue::get_uid)
            .def_readwrite("value", &ProdSimplexValue::value_)
            .def("dim", &ProdSimplexValue::dim)
            .def("boundary", &ProdSimplexValue::boundary)
            .def("__repr__", [](const ProdSimplexValue& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    py::class_<Filtration>(m, filtration_class_name.c_str())
            .def(py::init<typename Filtration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate") = false,
                    py::arg("n_threads") = 1,
                    py::arg("sort_only_by_dimension") = false,
                    py::arg("set_ids") = true)
            .def("max_dim", &Filtration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &Filtration::cells_copy, "copy of all cells in filtration order")
            .def("simplices", &Filtration::cells_copy, "copy of all simplices (cells) in filtration order")
            .def("size", &Filtration::size, "number of cells in filtration")
            .def("__len__", &Filtration::size)
            .def("size_in_dimension", &Filtration::size_in_dimension, py::arg("dim"), "number of cells of dimension dim")
            .def("n_vertices", &Filtration::n_vertices)
            .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, py::arg("sorted_id"))
            .def("get_id_by_sorted_id", &Filtration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("get_sorted_id_by_id", &Filtration::get_sorted_id, py::arg("id"))
            .def("get_cell", &Filtration::get_cell, py::arg("i"))
            .def("get_simplex", &Filtration::get_cell, py::arg("i"))
            .def("get_sorting_permutation", &Filtration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .def("simplex_value_by_vertices", &Filtration::value_by_vertices, py::arg("vertices"))
            .def("get_sorted_id_by_vertices", &Filtration::get_sorted_id_by_vertices, py::arg("vertices"))
            .def("cell_by_uid", &Filtration::get_cell_by_uid, py::arg("uid"))
            .def("boundary_matrix", &Filtration::boundary_matrix_full)
            .def("coboundary_matrix", &Filtration::coboundary_matrix)
            .def("boundary_matrix_rel", &Filtration::boundary_matrix_full_rel)
            .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            .def("set_values", &Filtration::set_values)
            .def("__iter__", [](Filtration& fil) { return py::make_iterator(fil.begin(), fil.end()); }, py::keep_alive<0, 1>())
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
            .def(py::init<typename ProdFiltration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate") = false,
                    py::arg("n_threads") = 1,
                    py::arg("sort_only_by_dimension") = false,
                    py::arg("set_ids") = true)
            .def("max_dim", &ProdFiltration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &ProdFiltration::cells_copy, "copy of all cells in filtration order")
            .def("size", &ProdFiltration::size, "number of cells in filtration")
            .def("__len__", &ProdFiltration::size)
            .def("__iter__", [](ProdFiltration& fil) { return py::make_iterator(fil.begin(), fil.end()); }, py::keep_alive<0, 1>())
            .def("size_in_dimension", &ProdFiltration::size_in_dimension, py::arg("dim"), "number of cells of dimension dim")
            .def("cell_value_by_sorted_id", &ProdFiltration::value_by_sorted_id, py::arg("sorted_id"))
            .def("get_id_by_sorted_id", &ProdFiltration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("get_sorted_id_by_id", &ProdFiltration::get_sorted_id, py::arg("id"))
            .def("get_cell", &ProdFiltration::get_cell, py::arg("i"))
            .def("get_sorting_permutation", &ProdFiltration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &ProdFiltration::get_inv_sorting_permutation)
            .def("boundary_matrix", &ProdFiltration::boundary_matrix_full)
            .def("reset_ids_to_sorted_ids", &ProdFiltration::reset_ids_to_sorted_ids)
            .def("set_values", &ProdFiltration::set_values)
            .def("__repr__", [](const ProdFiltration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            });

    m.def("mapping_cylinder", &oin::build_mapping_cylinder<Simplex, oin_real>, py::arg("fil_domain"), py::arg("fil_codomain"), py::arg("v_domain"), py::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("mapping_cylinder_with_indices", &oin::build_mapping_cylinder_with_indices<Simplex, oin_real>, py::arg("fil_domain"), py::arg("fil_codomain"), py::arg("v_domain"), py::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("multiply_filtration", &oin::multiply_filtration<Simplex, oin_real>, py::arg("fil"), py::arg("sigma"), "return a filtration with each simplex in fil multiplied by simplex sigma");
//    m.def("kernel_diagrams", &compute_kernel_diagrams<ProdSimplex, oin_real>, py::arg("fil_L"), py::arg("fil_K"), "return kernel persistence diagrams of inclusion fil_L -> fil_K");
//    m.def("kernel_cokernel_diagrams", &compute_kernel_cokernel_diagrams<ProdSimplex, oin_real>, py::arg("fil_L"), py::arg("fil_K"), "return kernel and cokernel persistence diagrams of inclusion fil_L -> fil_K");

    m.def("min_filtration", &oin::min_filtration<Simplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a filtration where each simplex has minimal value from fil_1, fil_2");
    m.def("min_filtration", &oin::min_filtration<ProdSimplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a filtration where each cell has minimal value from fil_1, fil_2");

    // helper for differentiable filtration
    m.def("min_filtration_with_indices", &oin::min_filtration_with_indices<Simplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
    m.def("min_filtration_with_indices", &oin::min_filtration_with_indices<ProdSimplex, oin_real>, py::arg("fil_1"), py::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");

    py::class_<KerImCokRedSimplex>(m, ker_im_cok_reduced_class_name.c_str())
            .def(py::init<const Filtration&, const Filtration&, oin::KICRParams&>(), py::arg("K"), py::arg("L"), py::arg("params"))
            .def("domain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_domain_diagrams()); })
            .def("codomain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_codomain_diagrams()); })
            .def("kernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_readwrite("decomposition_f", &KerImCokRedSimplex::dcmp_F_)
            .def_readwrite("decomposition_g", &KerImCokRedSimplex::dcmp_G_)
            .def_readwrite("decomposition_im", &KerImCokRedSimplex::dcmp_im_)
            .def_readwrite("decomposition_ker", &KerImCokRedSimplex::dcmp_ker_)
            .def_readwrite("decomposition_cok", &KerImCokRedSimplex::dcmp_cok_)
            ;

     py::class_<KerImCokRedProdSimplex>(m, ker_im_cok_reduced_prod_class_name.c_str())
            .def(py::init<const ProdFiltration&, const ProdFiltration&, oin::KICRParams&>(), py::arg("K"), py::arg("L"), py::arg("params"))
            .def("kernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_readwrite("decomposition_f", &KerImCokRedProdSimplex::dcmp_F_)
            .def_readwrite("decomposition_g", &KerImCokRedProdSimplex::dcmp_G_)
            .def_readwrite("decomposition_im", &KerImCokRedProdSimplex::dcmp_im_)
            .def_readwrite("decomposition_ker", &KerImCokRedProdSimplex::dcmp_ker_)
            .def_readwrite("decomposition_cok", &KerImCokRedProdSimplex::dcmp_cok_)
            ;
}