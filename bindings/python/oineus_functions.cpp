#include "oineus_persistence_bindings.h"
#include <hera/bottleneck.h>
#include <hera/wasserstein.h>

namespace {

using DiagramPoint = oin::Diagrams<oin_real>::Point;
using Diagram = oin::Diagrams<oin_real>::Dgm;
using DiagramCollection = PyOineusDiagrams<oin_real>;
using NumpyDiagram = nb::ndarray<oin_real, nb::c_contig, nb::device::cpu, nb::ro>;

Diagram normalize_diagram_ids(const Diagram& dgm)
{
    Diagram result = dgm;
    for(size_t i = 0 ; i < result.size() ; ++i) {
        result[i].id = static_cast<oin::id_type>(i);
    }
    return result;
}

Diagram diagrams_to_diagram(const DiagramCollection& diagrams, dim_type dim)
{
    try {
        return normalize_diagram_ids(diagrams.data().get_diagram_in_dimension(dim));
    } catch (const std::out_of_range&) {
        throw nb::index_error("Diagram dimension out of range");
    }
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

nb::ndarray<oin_real, nb::numpy> diagram_to_numpy(const Diagram& dgm)
{
    size_t arr_sz = dgm.size() * 2;
    auto* ptr = new oin_real[arr_sz];
    for(size_t i = 0; i < dgm.size(); ++i) {
        ptr[2 * i] = dgm[i].birth;
        ptr[2 * i + 1] = dgm[i].death;
    }

    nb::capsule free_when_done(ptr, [](void* p) noexcept {
        auto* pp = reinterpret_cast<oin_real*>(p);
        delete[] pp;
    });

    return nb::ndarray<oin_real, nb::numpy>(ptr, {dgm.size(), static_cast<size_t>(2)}, free_when_done);
}

Diagram python_object_to_diagram(const nb::handle& obj)
{
    try {
        return numpy_to_diagram(nb::cast<NumpyDiagram>(obj));
    } catch (const nb::cast_error&) {
    }

    try {
        return nb::cast<Diagram>(obj);
    } catch (const nb::cast_error&) {
    }

    throw nb::type_error("Expected a persistence diagram as list[DiagramPoint] or NumPy array with shape (n_points, 2)");
}

std::vector<Diagram> python_object_to_diagrams(const nb::list& diagrams)
{
    std::vector<Diagram> result;
    result.reserve(nb::len(diagrams));

    for (auto item : diagrams)
        result.push_back(python_object_to_diagram(item));

    return result;
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
            nb::call_guard<nb::gil_scoped_release>(),
            "Compute bottleneck distance between two persistence diagrams.");

    m.def(func_name.c_str(),
            [](const DiagramCollection& dgm_1, const DiagramCollection& dgm_2, dim_type dim, oin_real delta) {
                return bottleneck_distance_impl(diagrams_to_diagram(dgm_1, dim), diagrams_to_diagram(dgm_2, dim), delta);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("dim"), nb::arg("delta") = 0.01,
            nb::call_guard<nb::gil_scoped_release>(),
            "Compute bottleneck distance between two Oineus Diagrams objects in a fixed homology dimension.");

    m.def(func_name.c_str(),
            [](const NumpyDiagram& dgm_1, const NumpyDiagram& dgm_2, oin_real delta) {
                return bottleneck_distance_impl(numpy_to_diagram(dgm_1), numpy_to_diagram(dgm_2), delta);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("delta") = 0.01,
            nb::call_guard<nb::gil_scoped_release>(),
            "Compute bottleneck distance between two persistence diagrams given as NumPy arrays of shape (n_points, 2).");

    func_name = "wasserstein_distance";
    m.def(func_name.c_str(),
            [](const Diagram& dgm_1, const Diagram& dgm_2, oin_real q, oin_real delta, oin_real internal_p) {
                return wasserstein_distance_impl(normalize_diagram_ids(dgm_1), normalize_diagram_ids(dgm_2), q, delta, internal_p);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("q") = 1.0,
            nb::arg("delta") = 0.01, nb::arg("internal_p") = hera::get_infinity<oin_real>(),
            nb::call_guard<nb::gil_scoped_release>(),
            "Compute q-Wasserstein distance between two persistence diagrams.");

    m.def(func_name.c_str(),
            [](const DiagramCollection& dgm_1, const DiagramCollection& dgm_2, dim_type dim, oin_real q, oin_real delta, oin_real internal_p) {
                return wasserstein_distance_impl(diagrams_to_diagram(dgm_1, dim), diagrams_to_diagram(dgm_2, dim), q, delta, internal_p);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("dim"), nb::arg("q") = 1.0,
            nb::arg("delta") = 0.01, nb::arg("internal_p") = hera::get_infinity<oin_real>(),
            nb::call_guard<nb::gil_scoped_release>(),
            "Compute q-Wasserstein distance between two Oineus Diagrams objects in a fixed homology dimension.");

    m.def(func_name.c_str(),
            [](const NumpyDiagram& dgm_1, const NumpyDiagram& dgm_2, oin_real q, oin_real delta, oin_real internal_p) {
                return wasserstein_distance_impl(numpy_to_diagram(dgm_1), numpy_to_diagram(dgm_2), q, delta, internal_p);
            },
            nb::arg("dgm_1"), nb::arg("dgm_2"), nb::arg("q") = 1.0,
            nb::arg("delta") = 0.01, nb::arg("internal_p") = hera::get_infinity<oin_real>(),
            nb::call_guard<nb::gil_scoped_release>(),
            "Compute q-Wasserstein distance between NumPy-array persistence diagrams of shape (n_points, 2).");

    func_name = "frechet_mean";
    m.def(func_name.c_str(),
            [](const nb::list& diagrams,
               nb::object weights,
               size_t max_iter,
               oin_real tol,
               oin_real wasserstein_delta,
               oin_real internal_p,
               oin::FrechetMeanInit init_strategy,
               oin::DiagramPlaneDomain domain,
               bool ignore_infinite_points,
               oin_real random_noise_scale,
               size_t random_seed,
               size_t grid_n_x_bins,
               size_t grid_n_y_bins,
               nb::object custom_initial_barycenter) {
                auto diagram_vec = python_object_to_diagrams(diagrams);
                std::vector<oin_real> weight_vec;
                if (!weights.is_none()) {
                    nb::sequence weight_seq = nb::cast<nb::sequence>(weights);
                    weight_vec.reserve(nb::len(weight_seq));
                    for (auto item : weight_seq)
                        weight_vec.push_back(nb::cast<oin_real>(item));
                }

                oin::FrechetMeanParams<oin_real> params;
                params.max_iter = max_iter;
                params.tol = tol;
                params.wasserstein_delta = wasserstein_delta;
                params.internal_p = (internal_p < 0 || std::isinf(internal_p)) ? std::numeric_limits<oin_real>::infinity() : internal_p;
                params.init_strategy = init_strategy;
                params.domain = domain;
                params.ignore_infinite_points = ignore_infinite_points;
                params.random_init_params.noise_scale = random_noise_scale;
                params.random_init_params.random_seed = random_seed;
                params.random_init_params.domain = domain;
                params.grid_init_params.n_x_bins = std::max<size_t>(1, grid_n_x_bins);
                params.grid_init_params.n_y_bins = std::max<size_t>(1, grid_n_y_bins);
                params.grid_init_params.domain = domain;

                Diagram custom_dgm;
                if (!custom_initial_barycenter.is_none())
                    custom_dgm = python_object_to_diagram(custom_initial_barycenter);

                auto result = oin::frechet_mean<oin_real>(diagram_vec, weight_vec, params, custom_dgm);
                return diagram_to_numpy(result);
            },
            nb::arg("diagrams"),
            nb::arg("weights") = nb::none(),
            nb::arg("max_iter") = 100,
            nb::arg("tol") = 1e-7,
            nb::arg("wasserstein_delta") = 0.01,
            nb::arg("internal_p") = hera::get_infinity<oin_real>(),
            nb::arg("init_strategy") = oin::FrechetMeanInit::Grid,
            nb::arg("domain") = oin::DiagramPlaneDomain::AboveDiagonal,
            nb::arg("ignore_infinite_points") = false,
            nb::arg("random_noise_scale") = 1.0,
            nb::arg("random_seed") = 42,
            nb::arg("grid_n_x_bins") = 16,
            nb::arg("grid_n_y_bins") = 16,
            nb::arg("custom_initial_barycenter") = nb::none(),
            "Compute a Frechet mean (W2 barycenter) of persistence diagrams.");
}
