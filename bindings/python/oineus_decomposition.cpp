#include "oineus_persistence_bindings.h"

void init_oineus_common_decomposition(py::module& m)
{
    using namespace pybind11::literals;

    using Decomposition = oin::VRUDecomposition<oin_int>;
    using Simplex = oin::Simplex<oin_int>;
    using SimplexFiltration = oin::Filtration<Simplex, oin_real>;
    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexFiltration = oin::Filtration<ProdSimplex, oin_real>;

    py::class_<Decomposition>(m, "Decomposition")
            .def(py::init<const SimplexFiltration&, bool, int>(), py::arg("filtration"), py::arg("dualize"), py::arg("n_threads")=4)
            .def(py::init<const ProdSimplexFiltration&, bool, int>(), py::arg("filtration"), py::arg("dualize"), py::arg("n_threads")=4)
            .def(py::init<const typename Decomposition::MatrixData&, size_t, bool, bool>(),
                    py::arg("d"), py::arg("n_rows"), py::arg("dualize")=false, py::arg("skip_check")=false)
            .def_readwrite("r_data", &Decomposition::r_data)
            .def_readwrite("v_data", &Decomposition::v_data)
            .def_readwrite("u_data_t", &Decomposition::u_data_t)
            .def_readwrite("d_data", &Decomposition::d_data)
            .def("reduce", &Decomposition::reduce, py::arg("params")=oin::Params(), py::call_guard<py::gil_scoped_release>())
            .def("sanity_check", &Decomposition::sanity_check, py::call_guard<py::gil_scoped_release>())
            .def("diagram", [](const Decomposition& self, const SimplexFiltration& fil, bool include_inf_points)
                            { return PyOineusDiagrams<oin_real>(self.diagram(fil, include_inf_points)); },
                    py::arg("fil"), py::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const ProdSimplexFiltration& fil, bool include_inf_points) { return PyOineusDiagrams<oin_real>(self.diagram(fil,
                    include_inf_points)); },
                    py::arg("fil"), py::arg("include_inf_points") = true)
            .def("zero_pers_diagram", [](const Decomposition& self, const SimplexFiltration& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    py::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const ProdSimplexFiltration& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    py::arg("fil"))
            .def("filtration_index", &Decomposition::filtration_index, py::arg("matrix_index"))
                    ;

}