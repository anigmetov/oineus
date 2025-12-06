#include "oineus_persistence_bindings.h"

void init_oineus_common_decomposition(nb::module_& m)
{
    using Decomposition = oin::VRUDecomposition<oin_int>;
    using Simplex = oin::Simplex<oin_int>;
    using SimplexFiltration = oin::Filtration<Simplex, oin_real>;
    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexFiltration = oin::Filtration<ProdSimplex, oin_real>;
    using CubeFiltration_1D = oin::Filtration<oin::Cube<oin_int, 1>, oin_real>;
    using CubeFiltration_2D = oin::Filtration<oin::Cube<oin_int, 2>, oin_real>;
    using CubeFiltration_3D = oin::Filtration<oin::Cube<oin_int, 3>, oin_real>;

    nb::class_<Decomposition>(m, "Decomposition")
            .def(nb::init<const SimplexFiltration&, bool, int>(), nb::arg("filtration"), nb::arg("dualize"), nb::arg("n_threads")=4)
            .def(nb::init<const ProdSimplexFiltration&, bool, int>(), nb::arg("filtration"), nb::arg("dualize"), nb::arg("n_threads")=4)
            .def(nb::init<const CubeFiltration_1D&, bool, int>(), nb::arg("filtration"), nb::arg("dualize"), nb::arg("n_threads")=4)
            .def(nb::init<const CubeFiltration_2D&, bool, int>(), nb::arg("filtration"), nb::arg("dualize"), nb::arg("n_threads")=4)
            .def(nb::init<const CubeFiltration_3D&, bool, int>(), nb::arg("filtration"), nb::arg("dualize"), nb::arg("n_threads")=4)
            .def(nb::init<const typename Decomposition::MatrixData&, size_t, bool, bool>(),
                    nb::arg("d"), nb::arg("n_rows"), nb::arg("dualize")=false, nb::arg("skip_check")=false)
            .def_rw("r_data", &Decomposition::r_data)
            .def_rw("v_data", &Decomposition::v_data)
            .def_rw("u_data_t", &Decomposition::u_data_t)
            .def_ro("d_data", &Decomposition::d_data)
            .def_prop_ro("dualize", &Decomposition::dualize)
            .def("reduce", &Decomposition::reduce, nb::arg("params")=oin::Params(), nb::call_guard<nb::gil_scoped_release>())
            .def("is_elz", &Decomposition::is_elz, nb::arg("n_threads")=8, nb::call_guard<nb::gil_scoped_release>())
            .def("n_elz_violators", &Decomposition::n_elz_violators, nb::arg("n_threads")=8, nb::call_guard<nb::gil_scoped_release>())
            .def("n_elz_violators_in_dim", &Decomposition::n_elz_violators_in_dim, nb::arg("dim"), nb::arg("n_threads")=8, nb::call_guard<nb::gil_scoped_release>())
            .def("is_column_elz", &Decomposition::is_column_elz, nb::arg("column_idx"))
            .def("restore_elz", &Decomposition::restore_elz)
            .def("densify_v_for_selinv", &Decomposition::densify_v_for_selinv, nb::arg("rows_to_invert"))
            .def("sanity_check", &Decomposition::sanity_check, nb::call_guard<nb::gil_scoped_release>())
            .def("diagram", [](const Decomposition& self, const SimplexFiltration& fil, bool include_inf_points)
                            { return PyOineusDiagrams<oin_real>(self.diagram(fil, include_inf_points)); },
                    nb::arg("fil"), nb::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const ProdSimplexFiltration& fil, bool include_inf_points) { return PyOineusDiagrams<oin_real>(self.diagram(fil,
                    include_inf_points)); },
                    nb::arg("fil"), nb::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const CubeFiltration_1D& fil, bool include_inf_points) { return PyOineusDiagrams<oin_real>(self.diagram(fil,
                    include_inf_points)); },
                    nb::arg("fil"), nb::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const CubeFiltration_2D& fil, bool include_inf_points) { return PyOineusDiagrams<oin_real>(self.diagram(fil,
                    include_inf_points)); },
                    nb::arg("fil"), nb::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const CubeFiltration_3D& fil, bool include_inf_points) { return PyOineusDiagrams<oin_real>(self.diagram(fil,
                    include_inf_points)); },
                    nb::arg("fil"), nb::arg("include_inf_points") = true)
            .def("zero_pers_diagram", [](const Decomposition& self, const SimplexFiltration& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    nb::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const ProdSimplexFiltration& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    nb::arg("fil"))
            .def("filtration_index", &Decomposition::filtration_index, nb::arg("matrix_index"))
                    ;

}