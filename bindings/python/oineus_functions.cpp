#include "oineus_persistence_bindings.h"

void init_oineus_functions(py::module& m)
{
    using namespace pybind11::literals;

    using Simp = oin::Simplex<oin_int>;
    using SimpProd = oin::ProductCell<Simp, Simp>;
    using Filtration = oin::Filtration<Simp, oin_real>;

    using oin::VREdge;

    std::string func_name;

    // diagrams
    func_name = "compute_diagrams_ls";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<oin_int, oin_real>);

    // Lower-star Freudenthal filtration
    func_name = "get_freudenthal_filtration";
    m.def(func_name.c_str(), &get_fr_filtration<oin_int, oin_real>,
            py::arg("data"), py::arg("negate") = false, py::arg("wrap") = false, py::arg("max_dim") = 3, py::arg("n_threads") = 1);

    func_name = "get_freudenthal_filtration_and_crit_vertices";
    m.def(func_name.c_str(), &get_fr_filtration_and_critical_vertices<oin_int, oin_real>,
            py::arg("data"), py::arg("negate") = false, py::arg("wrap") = false, py::arg("max_dim") = 3, py::arg("n_threads") = 1);

    // Vietoris--Rips filtration
    // Reasonable default (dimension of points) for max_dim is provided in Python
    // in C++, max_diameter is +\infty by default
    // Reasonable default (diameter of point cloud)
    // is provided in __init__.py on pure Python level
    func_name = "get_vr_filtration";
    m.def(func_name.c_str(), &get_vr_filtration<oin_int, oin_real>,
            py::arg("points"), py::arg("max_dim"), py::arg("max_diameter")=std::numeric_limits<oin_real>::max(), py::arg("n_threads")=1);

    func_name = "get_vr_filtration_and_critical_edges";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges<oin_int, oin_real>,
            py::arg("points"), py::arg("max_dim"), py::arg("max_diameter")=std::numeric_limits<oin_real>::max(), py::arg("n_threads")=1);

    func_name = "get_vr_filtration_from_pwdists";
    m.def(func_name.c_str(), &get_vr_filtration_from_pwdists<oin_int, oin_real>,
            py::arg("pwdists"), py::arg("max_dim"), py::arg("max_diameter")=std::numeric_limits<oin_real>::max(), py::arg("n_threads")=1);

    func_name = "get_vr_filtration_and_critical_edges_from_pwdists";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges_from_pwdists<oin_int, oin_real>,
            py::arg("pwdists"), py::arg("max_dim"), py::arg("max_diameter")=std::numeric_limits<oin_real>::max(), py::arg("n_threads")=1);

    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix";
    m.def(func_name.c_str(), &get_boundary_matrix<oin_int, oin_real>);

    // target values
    func_name = "get_denoise_target";
    m.def(func_name.c_str(), &oin::get_denoise_target<Simp, oin_real>);

    // target values -- diagram loss
    func_name = "get_target_values_diagram_loss";
    m.def(func_name.c_str(), &oin::get_prescribed_simplex_values_diagram_loss<oin_real>);

    // target values --- X set
    func_name = "get_target_values_x";
    m.def(func_name.c_str(), &oin::get_prescribed_simplex_values_set_x<Simp, oin_real>);

    // to reproduce "Well group loss" experiments
    func_name = "get_well_group_target";
    m.def(func_name.c_str(), &oin::get_well_group_target<Simp, oin_real>);

    func_name = "get_nth_persistence";
    m.def(func_name.c_str(), &oin::get_nth_persistence<Simp, oin_real>);

    // to get permutation for Warm Starts
    func_name = "get_permutation";
    m.def(func_name.c_str(), &oin::targets_to_permutation<Simp, oin_real>);

    func_name = "get_permutation_dtv";
    m.def(func_name.c_str(), &oin::targets_to_permutation_dtv<Simp, oin_real>);

    func_name = "list_to_filtration";
    m.def(func_name.c_str(), &list_to_filtration<oin_int, oin_real>);

    func_name = "compute_kernel_image_cokernel_reduction";
    m.def(func_name.c_str(), &compute_kernel_image_cokernel_reduction<Simp, oin_real>);

    func_name = "get_ls_filtration";
    m.def(func_name.c_str(), &get_ls_filtration<oin_int, oin_real>);

    func_name = "compute_relative_diagrams";
    m.def(func_name.c_str(), &compute_relative_diagrams<Simp, oin_real>, py::arg("fil"), py::arg("rel"), py::arg("include_inf_points")=true);

    func_name = "compute_relative_diagrams";
    m.def(func_name.c_str(), &compute_relative_diagrams<SimpProd, oin_real>, py::arg("fil"), py::arg("rel"), py::arg("include_inf_points")=true);
}
