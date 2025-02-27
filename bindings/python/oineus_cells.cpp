#include "oineus_persistence_bindings.h"

void init_oineus_cells(py::module& m)
{
    using namespace pybind11::literals;

    using Simplex = oin::Simplex<oin_int>;
    using SimplexValue = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexValue = oin::CellWithValue<ProdSimplex, oin_real>;

    // Simplex and product of two simplices without value, vertices only
    const std::string pure_simplex_class_name = "CombinatorialSimplex";
    const std::string pure_prod_simplex_class_name = "CombinatorialProdSimplex";

    const std::string simplex_class_name = "Simplex";
    const std::string prod_simplex_class_name = "ProdSimplex";

    py::class_<Simplex>(m, pure_simplex_class_name.c_str())
        .def(py::init([](Simplex::IdxVector vs) -> Simplex { return Simplex(vs, true); }), py::arg("vertices"))
        .def(py::init([](oin_int id, Simplex::IdxVector vs) -> Simplex { return Simplex(id, vs, true); }), py::arg("id"), py::arg("vertices"))
        .def("__iter__", [](Simplex& sigma) { return py::make_iterator(sigma.vertices_.begin(), sigma.vertices_.end()); }, py::keep_alive<0, 1>())
        .def("__getitem__", [](Simplex& sigma, size_t i) { return sigma.vertices_[i]; })
        .def_property("id", &Simplex::get_id, &Simplex::set_id)
        .def_readonly("vertices", &Simplex::vertices_)
        .def_property_readonly("uid", &Simplex::get_uid)
        .def("dim", &Simplex::dim)
        .def("boundary", &Simplex::boundary)
        .def("join", [](const Simplex& sigma, oin_int new_vertex, oin_int new_id) {
                  return sigma.join(new_id, new_vertex);
                },
                py::arg("new_vertex"),
                py::arg("new_id") = Simplex::k_invalid_id)
        .def("__repr__", [](const Simplex& sigma) {
          std::stringstream ss;
          ss << sigma;
          return ss.str();
        });

    py::class_<ProdSimplex>(m, pure_prod_simplex_class_name.c_str())
        .def(py::init([](Simplex::IdxVector vs1, Simplex::IdxVector vs2) -> ProdSimplex
            { return ProdSimplex(Simplex(vs1, true), Simplex(vs2, true)); }), py::arg("vertices_1"), py::arg("vertices_2"))
        .def_property("id", &ProdSimplex::get_id, &ProdSimplex::set_id)
        .def_property_readonly("factor_1", &ProdSimplex::get_factor_1)
        .def_property_readonly("factor_2", &ProdSimplex::get_factor_2)
        .def_property_readonly("uid", &ProdSimplex::get_uid)
        .def("dim", &ProdSimplex::dim)
        .def("boundary", &ProdSimplex::boundary)
        .def("__repr__", [](const ProdSimplex& sigma) {
          std::stringstream ss;
          ss << sigma;
          return ss.str();
        });

    py::class_<SimplexValue>(m, simplex_class_name.c_str())
            .def(py::init([](Simplex::IdxVector vs, oin_real value) -> SimplexValue {
                      return SimplexValue({vs}, value);
                    }),
                    py::arg("vertices"),
                    py::arg("value")=0.0)
            .def(py::init([](oin_int id, Simplex::IdxVector vs, oin_real value) -> SimplexValue { return SimplexValue({id, vs}, value); }), py::arg("id"), py::arg("vertices"), py::arg("value"))
            .def("__iter__", [](SimplexValue& sigma) { return py::make_iterator(sigma.cell_.vertices_.begin(), sigma.cell_.vertices_.end()); }, py::keep_alive<0, 1>())
            .def("__getitem__", [](SimplexValue& sigma, size_t i) { return sigma.cell_.vertices_[i]; })
            .def_property("id", &SimplexValue::get_id, &SimplexValue::set_id)
            .def_readwrite("sorted_id", &SimplexValue::sorted_id_)
            .def_property_readonly("vertices", [](const SimplexValue& sigma) { return sigma.cell_.vertices_; })
            .def_property_readonly("uid", &SimplexValue::get_uid)
            .def_readwrite("value", &SimplexValue::value_)
            .def("dim", &SimplexValue::dim)
            .def("boundary", &SimplexValue::boundary)
            .def("combinatorial_simplex", &SimplexValue::get_cell)
            .def("combinatorial_cell", &SimplexValue::get_cell)
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
            .def("combinatorial_cell", &ProdSimplexValue::get_cell)
            .def("__repr__", [](const ProdSimplexValue& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });
}