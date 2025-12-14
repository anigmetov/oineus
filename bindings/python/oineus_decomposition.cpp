#include "oineus_persistence_bindings.h"
#include <Eigen/SparseCore>
#include "nanobind/eigen/sparse.h"
#include "nanobind/stl/set.h"
#include "nanobind/stl/list.h"


Eigen::SparseMatrix<oin_int, Eigen::RowMajor> densify_v_for_selinv(
    const oin::VRUDecomposition<oin_int>& dcmp,
    const std::set<oin_int>& rows_to_invert,
    int num_rows_, int n_threads=1 )
{
    Timer timer;
    if (not dcmp.has_matrix_v())
        throw std::runtime_error("densify_v_for_selinv called on matrix without V");
    if (not dcmp.is_reduced)
        throw std::runtime_error("densify_v_for_selinv called on unreduced decomposition");

    const size_t num_rows = num_rows_;

    // Helper lambda to find dimension index for a given row
    auto get_dim = [&dcmp](oin_int idx) -> dim_type {
        auto it = std::upper_bound(dcmp.dim_last.begin(), dcmp.dim_last.end(), idx);
        return static_cast<dim_type>(it - dcmp.dim_last.begin());
    };

    Eigen::SparseMatrix<oin_int, Eigen::RowMajor> densified_v(num_rows, num_rows);

    if (dcmp.v_data.empty()) {
        return densified_v;
    }

    // count non-zeros per row
    std::vector<int> row_sizes(num_rows, 0);

    // First pass: count non-zeros from all columns
    for (size_t col_idx = 0; col_idx < dcmp.v_data.size(); ++col_idx) {
        for (auto row_idx : dcmp.v_data[col_idx]) {
            row_sizes[row_idx]++;
        }
    }

    // Second pass: override for densified rows
    for (auto row_idx : rows_to_invert) {
        dim_type d = get_dim(row_idx);
        // From diagonal (row_idx) to dim_last[d], inclusive
        row_sizes[row_idx] = dcmp.dim_last[d] - row_idx + 1;
    }

    densified_v.reserve(row_sizes);

    // Process each column
    for (size_t col_idx = 0; col_idx < dcmp.v_data.size(); ++col_idx) {
        const auto& column = dcmp.v_data[col_idx];

        // For each row entry in this column, decide how to insert
        std::set<oin_int> column_set(column.begin(), column.end());

        for (int row_idx : column) {
            if (rows_to_invert.find(row_idx) == rows_to_invert.end()) {
                // Normal row: just insert 1
                densified_v.insert(row_idx, col_idx) = 1;
            }
            // If row is in rows_to_invert, we handle it separately below
        }

        // Handle densified rows: for each row in rows_to_invert that could have
        // entries in this column (i.e., col_idx is within [row_idx, dim_last[d]])
        for (auto row_idx : rows_to_invert) {
            dim_type d = get_dim(row_idx);
            // Only process if col_idx is in the range [row_idx, dim_last[d]]
            if (col_idx >= static_cast<size_t>(row_idx) && col_idx <= static_cast<size_t>(dcmp.dim_last[d])) {
                if (column_set.find(row_idx) != column_set.end()) {
                    densified_v.insert(row_idx, col_idx) = 1;
                } else {
                    densified_v.insert(row_idx, col_idx) = 2;  // Explicit zero placeholder
                }
            }
        }
    }

    densified_v.makeCompressed();

    // Replace all 2s with 0s in compressed storage
    auto* values = densified_v.valuePtr();
    for (int k = 0; k < densified_v.nonZeros(); ++k) {
        if (values[k] == 2) {
            values[k] = 0;
        }
    }

    std::cerr << "Densifying V took " << timer.elapsed() << " seconds" << std::endl;

    return densified_v;
}

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
            .def("densify_v_for_selinv", [](Decomposition& self, const std::set<oin_int>& rows_to_invert, int n_threads) -> Eigen::SparseMatrix<oin_int, Eigen::RowMajor> {
                     int num_rows = self.r_data.size();
                     return densify_v_for_selinv(self, rows_to_invert, num_rows, n_threads);
                 },
                 nb::arg("rows_to_invert"), nb::arg("n_threads")=1,
                 nb::call_guard<nb::gil_scoped_release>())
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
            .def("zero_pers_diagram", [](const Decomposition& self, const CubeFiltration_1D& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    nb::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const CubeFiltration_2D& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    nb::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const CubeFiltration_3D& fil) { return PyOineusDiagrams<oin_real>(self.zero_persistence_diagram(fil)); },
                    nb::arg("fil"))
            .def("filtration_index", &Decomposition::filtration_index, nb::arg("matrix_index"))
                    ;

}