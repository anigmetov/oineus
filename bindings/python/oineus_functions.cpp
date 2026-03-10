#include "oineus_persistence_bindings.h"
#include <hera/bottleneck.h>
#include <hera/wasserstein.h>

namespace {

using DiagramPoint = oin::Diagrams<oin_real>::Point;
using Diagram = oin::Diagrams<oin_real>::Dgm;
using NumpyDiagram = nb::ndarray<oin_real, nb::c_contig, nb::device::cpu, nb::ro>;

Diagram normalize_diagram_ids(const Diagram& dgm)
{
    Diagram result = dgm;
    for(size_t i = 0 ; i < result.size() ; ++i) {
        result[i].id = static_cast<oin::id_type>(i);
    }
    return result;
}

Diagram numpy_to_diagram(const NumpyDiagram& dgm)
{
    if (dgm.ndim() != 2)
        throw nb::value_error("Expected 2D NumPy array with shape (n_points, 2)");
    if (dgm.shape(1) != 2)
        throw nb::value_error("Expected NumPy array with shape (n_points, 2)");

    Diagram result;
    result.reserve(dgm.shape(0));

    const auto* pdata = static_cast<const oin_real*>(dgm.data());
    for(size_t i = 0 ; i < dgm.shape(0) ; ++i) {
        result.emplace_back(pdata[2 * i], pdata[2 * i + 1]);
        result.back().id = static_cast<oin::id_type>(i);
    }

    return result;
}

oin_real bottleneck_distance_impl(Diagram dgm_1, Diagram dgm_2, oin_real delta)
{
    if (delta == 0.0)
        return hera::bottleneckDistExact(dgm_1, dgm_2);
    else
        return hera::bottleneckDistApprox(dgm_1, dgm_2, delta);
}

oin_real wasserstein_distance_impl(Diagram dgm_1, Diagram dgm_2, oin_real q, oin_real delta, oin_real internal_p)
{
    hera::AuctionParams<oin_real> params;
    params.wasserstein_power = q;
    params.delta = delta;
    params.internal_p = internal_p;
    return hera::wasserstein_dist(dgm_1, dgm_2, params);
}

} // namespace

void init_oineus_functions(nb::module_& m)
{
    using Simp = oin::Simplex<oin_int>;
    using SimpProd = oin::ProductCell<Simp, Simp>;
    using oin::VREdge;

    std::string func_name;

    // Lower-star Freudenthal filtration
    func_name = "get_freudenthal_filtration";
    m.def(func_name.c_str(), &get_fr_filtration<oin_int, oin_real>,
            nb::arg("data"), nb::arg("negate") = false, nb::arg("wrap") = false, nb::arg("max_dim") = 3, nb::arg("n_threads") = 1);

    func_name = "get_freudenthal_filtration_and_crit_vertices";
    m.def(func_name.c_str(), &get_fr_filtration_and_critical_vertices<oin_int, oin_real>,
            nb::arg("data"), nb::arg("negate") = false, nb::arg("wrap") = false, nb::arg("max_dim") = 3, nb::arg("n_threads") = 1);

    // Vietoris--Rips filtration
    // Reasonable default (dimension of points) for max_dim is provided in Python
    // in C++, max_diameter is +\infty by default
    // Reasonable default (diameter of point cloud)
    // is provided in __init__.py on pure Python level
    func_name = "get_vr_filtration";
    m.def(func_name.c_str(), &get_vr_filtration<oin_int, oin_real>,
            nb::arg("points"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    func_name = "get_vr_filtration_and_critical_edges";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges<oin_int, oin_real>,
            nb::arg("points"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    func_name = "get_vr_filtration_from_pwdists";
    m.def(func_name.c_str(), &get_vr_filtration_from_pwdists<oin_int, oin_real>,
            nb::arg("pwdists"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    func_name = "get_vr_filtration_and_critical_edges_from_pwdists";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges_from_pwdists<oin_int, oin_real>,
            nb::arg("pwdists"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix";
    m.def(func_name.c_str(), &get_boundary_matrix<oin_int, oin_real>);

    // target values
    func_name = "get_denoise_target";
    m.def(func_name.c_str(), &oin::get_denoise_target<Simp, oin_real>);

    func_name = "get_nth_persistence";
    m.def(func_name.c_str(), &oin::get_nth_persistence<Simp, oin_real>);

    // to get permutation for Warm Starts
    func_name = "get_permutation";
    m.def(func_name.c_str(), &oin::targets_to_permutation<Simp, oin_real>);

    func_name = "get_permutation_dtv";
    m.def(func_name.c_str(), &oin::targets_to_permutation_dtv<Simp, oin_real>);

    func_name = "compute_relative_diagrams";
    m.def(func_name.c_str(), &compute_relative_diagrams<Simp, oin_real>, nb::arg("fil"), nb::arg("rel"), nb::arg("include_inf_points")=true);

    func_name = "compute_relative_diagrams";
    m.def(func_name.c_str(), &compute_relative_diagrams<SimpProd, oin_real>, nb::arg("fil"), nb::arg("rel"), nb::arg("include_inf_points")=true);

    // persistence diagram distances via Hera
    func_name = "bottleneck_distance";
    m.def(func_name.c_str(),
            [](const Diagram& dgm_1, const Diagram& dgm_2, oin_real delta) {
                return bottleneck_distance_impl(normalize_diagram_ids(dgm_1), normalize_diagram_ids(dgm_2), delta);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("delta") = 0.01,
            "Compute bottleneck distance between two persistence diagrams.");

    m.def(func_name.c_str(),
            [](const NumpyDiagram& dgm_1, const NumpyDiagram& dgm_2, oin_real delta) {
                return bottleneck_distance_impl(numpy_to_diagram(dgm_1), numpy_to_diagram(dgm_2), delta);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("delta") = 0.01,
            "Compute bottleneck distance between two persistence diagrams given as NumPy arrays of shape (n_points, 2).");

    func_name = "wasserstein_distance";
    m.def(func_name.c_str(),
            [](const Diagram& dgm_1, const Diagram& dgm_2, oin_real q, oin_real delta, oin_real internal_p) {
                return wasserstein_distance_impl(normalize_diagram_ids(dgm_1), normalize_diagram_ids(dgm_2), q, delta, internal_p);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("q") = 2.0,
            nb::arg("delta") = 0.01, nb::arg("internal_p") = hera::get_infinity<oin_real>(),
            "Compute q-Wasserstein distance between two persistence diagrams.");

    m.def(func_name.c_str(),
            [](const NumpyDiagram& dgm_1, const NumpyDiagram& dgm_2, oin_real q, oin_real delta, oin_real internal_p) {
                return wasserstein_distance_impl(numpy_to_diagram(dgm_1), numpy_to_diagram(dgm_2), q, delta, internal_p);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("q") = 2.0,
            nb::arg("delta") = 0.01, nb::arg("internal_p") = hera::get_infinity<oin_real>(),
            "Compute q-Wasserstein distance between NumPy-array persistence diagrams of shape (n_points, 2).");
}
