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

struct WassersteinMatchingFinite {
    // Finite-to-finite matching (parallel arrays)
    std::vector<int> a_to_b;      // indices in finite_dgm_a
    std::vector<int> b_from_a;    // corresponding indices in finite_dgm_b

    // Finite-to-diagonal matching
    std::vector<int> a_to_diag;   // indices in finite_dgm_a matched to diagonal
    std::vector<int> b_to_diag;   // indices in finite_dgm_b matched to diagonal
};

// A single edge in the bottleneck matching whose length equals the
// bottleneck distance (one of possibly many tied edges).
// Exactly one of idx_a / idx_b may be -1, indicating that endpoint is
// a diagonal projection of the other side.
struct BottleneckLongestEdge {
    int      idx_a;
    int      idx_b;
    oin_real length;
    oin_real a_x;
    oin_real a_y;
    oin_real b_x;
    oin_real b_y;
};

struct BottleneckMatchingFinite {
    // Full finite matching (same groupings as WassersteinMatchingFinite):
    std::vector<int> a_to_b;
    std::vector<int> b_from_a;
    std::vector<int> a_to_diag;
    std::vector<int> b_to_diag;
    // Every edge whose length equals `distance`:
    std::vector<BottleneckLongestEdge> longest_edges;
    oin_real distance { 0 };
};

std::vector<Diagram> python_object_to_diagrams(const nb::list& diagrams)
{
    std::vector<Diagram> result;
    result.reserve(nb::len(diagrams));

    for (auto item : diagrams)
        result.push_back(python_object_to_diagram(item));

    return result;
}

WassersteinMatchingFinite wasserstein_matching_finite_impl(
    const NumpyDiagram& finite_a_np,
    const NumpyDiagram& finite_b_np,
    oin_real wasserstein_q,
    oin_real wasserstein_delta,
    oin_real internal_p)
{
    // Convert NumPy diagrams to Hera format
    Diagram finite_a = numpy_to_diagram(finite_a_np);
    Diagram finite_b = numpy_to_diagram(finite_b_np);

    WassersteinMatchingFinite result;

    // Handle empty diagrams
    if (finite_a.empty() && finite_b.empty()) {
        return result;  // Both empty, return empty matching
    }

    // Check if diagrams are identical (or very close)
    if (finite_a.size() == finite_b.size()) {
        bool identical = true;
        constexpr oin_real eps = 1e-10;

        for (size_t i = 0; i < finite_a.size() && identical; ++i) {
            if (std::abs(finite_a[i].birth - finite_b[i].birth) > eps ||
                std::abs(finite_a[i].death - finite_b[i].death) > eps) {
                identical = false;
            }
        }

        if (identical) {
            // Diagrams are identical, match each point to itself
            for (size_t i = 0; i < finite_a.size(); ++i) {
                result.a_to_b.push_back(i);
                result.b_from_a.push_back(i);
            }
            return result;
        }
    }

    // Call Hera directly on the original diagrams
    // Hera will build the augmented diagrams internally
    hera::AuctionParams<oin_real> params;
    params.wasserstein_power = wasserstein_q;
    params.delta = wasserstein_delta;
    params.internal_p = internal_p;
    params.return_matching = true;
    params.match_inf_points = false;  // Finite points only!

    auto hera_res = hera::wasserstein_cost_detailed(finite_a, finite_b, params);

    // Parse Hera's matching result
    // Important: Sort by a_id to maintain parallel array correspondence
    std::vector<std::pair<int, int>> finite_matches;
    std::vector<int> a_diag_matches, b_diag_matches;

    for (const auto& [a_id, b_id] : hera_res.matching_a_to_b_) {
        if (a_id >= 0 && b_id >= 0) {
            finite_matches.emplace_back(a_id, b_id);
        } else if (a_id >= 0 && b_id < 0) {
            a_diag_matches.push_back(a_id);
        } else if (a_id < 0 && b_id >= 0) {
            b_diag_matches.push_back(b_id);
        }
    }

    // Sort finite matches by a_id to create proper parallel arrays
    std::sort(finite_matches.begin(), finite_matches.end());

    for (const auto& [a_id, b_id] : finite_matches) {
        result.a_to_b.push_back(a_id);
        result.b_from_a.push_back(b_id);
    }

    // Sort diagonal matches for consistency
    std::sort(a_diag_matches.begin(), a_diag_matches.end());
    std::sort(b_diag_matches.begin(), b_diag_matches.end());

    result.a_to_diag = std::move(a_diag_matches);
    result.b_to_diag = std::move(b_diag_matches);

    return result;
}

BottleneckMatchingFinite bottleneck_matching_finite_impl(
    const NumpyDiagram& finite_a_np,
    const NumpyDiagram& finite_b_np,
    oin_real delta)
{
    Diagram finite_a = numpy_to_diagram(finite_a_np);
    Diagram finite_b = numpy_to_diagram(finite_b_np);

    BottleneckMatchingFinite result;

    if (finite_a.empty() && finite_b.empty()) {
        return result;
    }

    // Fast path: identical diagrams.
    if (finite_a.size() == finite_b.size()) {
        bool identical = true;
        constexpr oin_real eps = 1e-10;
        for (size_t i = 0; i < finite_a.size() && identical; ++i) {
            if (std::abs(finite_a[i].birth - finite_b[i].birth) > eps ||
                std::abs(finite_a[i].death - finite_b[i].death) > eps) {
                identical = false;
            }
        }
        if (identical) {
            for (size_t i = 0; i < finite_a.size(); ++i) {
                result.a_to_b.push_back(static_cast<int>(i));
                result.b_from_a.push_back(static_cast<int>(i));
            }
            result.distance = 0;
            return result;
        }
    }

    hera::BottleneckResult<oin_real> hera_res = (delta == 0.0)
        ? hera::bottleneckDetailedExact(finite_a, finite_b)
        : hera::bottleneckDetailedApprox(finite_a, finite_b, delta);

    result.distance = hera_res.distance;

    // Parse the full matching into a_to_b / a_to_diag / b_to_diag groups.
    // Hera gives us pairs of DiagramPoint; each DiagramPoint carries a
    // user_tag that is the original index (for NORMAL) or -1 - other_side_tag
    // (for DIAG projections).
    std::vector<std::pair<int, int>> finite_matches;
    std::vector<int> a_diag_matches;
    std::vector<int> b_diag_matches;

    using DP = hera::DiagramPoint<oin_real>;
    auto a_original_idx = [](const DP& p) -> int {
        return p.is_normal() ? p.user_tag : -1;
    };
    auto b_original_idx = [](const DP& p) -> int {
        return p.is_normal() ? p.user_tag : -1;
    };

    // Note: Hera's oracle builds a complete matching over the combined
    // (original + projections) point set, so every edge returned by
    // get_edges() has a NORMAL endpoint somewhere; endpoints from side A
    // land in `first` and side B in `second` with high probability, but
    // addProjections may have shuffled ownership. We disambiguate via
    // user_tag sign convention: non-negative => original on that side;
    // negative => projection coming from the OTHER side.
    for (const auto& edge : hera_res.edges) {
        const DP& pa = edge.first;
        const DP& pb = edge.second;
        int ia = a_original_idx(pa);
        int ib = b_original_idx(pb);
        if (ia >= 0 && ib >= 0) {
            finite_matches.emplace_back(ia, ib);
        } else if (ia >= 0 && ib < 0) {
            a_diag_matches.push_back(ia);
        } else if (ia < 0 && ib >= 0) {
            b_diag_matches.push_back(ib);
        }
        // diag<->diag edges are already filtered out by get_edges().
    }

    std::sort(finite_matches.begin(), finite_matches.end());
    for (const auto& [ia, ib] : finite_matches) {
        result.a_to_b.push_back(ia);
        result.b_from_a.push_back(ib);
    }
    std::sort(a_diag_matches.begin(), a_diag_matches.end());
    std::sort(b_diag_matches.begin(), b_diag_matches.end());
    result.a_to_diag = std::move(a_diag_matches);
    result.b_to_diag = std::move(b_diag_matches);

    // Translate longest edges.
    for (const auto& edge : hera_res.longest_edges) {
        const DP& pa = edge.first;
        const DP& pb = edge.second;
        BottleneckLongestEdge le;
        le.idx_a = a_original_idx(pa);
        le.idx_b = b_original_idx(pb);
        le.a_x = pa.getRealX();
        le.a_y = pa.getRealY();
        le.b_x = pb.getRealX();
        le.b_y = pb.getRealY();
        // Edge length matches the Hera-internal convention: L_inf for
        // non-diagonal pairs, persistence for diag<->normal pairs.
        if (pa.is_diagonal() && pb.is_normal()) {
            le.length = pb.persistence_lp(hera::get_infinity<oin_real>());
        } else if (pa.is_normal() && pb.is_diagonal()) {
            le.length = pa.persistence_lp(hera::get_infinity<oin_real>());
        } else {
            le.length = hera::dist_l_inf(pa, pb);
        }
        result.longest_edges.push_back(le);
    }

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

    // Bind WassersteinMatchingFinite struct
    nb::class_<WassersteinMatchingFinite>(m, "WassersteinMatchingFinite",
            "Matching information for Wasserstein distance between two finite diagrams")
        .def_ro("a_to_b", &WassersteinMatchingFinite::a_to_b,
            "Indices in finite_dgm_a matched to finite points in dgm_b")
        .def_ro("b_from_a", &WassersteinMatchingFinite::b_from_a,
            "Corresponding indices in finite_dgm_b")
        .def_ro("a_to_diag", &WassersteinMatchingFinite::a_to_diag,
            "Indices in finite_dgm_a matched to diagonal")
        .def_ro("b_to_diag", &WassersteinMatchingFinite::b_to_diag,
            "Indices in finite_dgm_b matched to diagonal");

    // Bind wasserstein_matching_finite function
    func_name = "wasserstein_matching_finite";
    m.def(func_name.c_str(),
        &wasserstein_matching_finite_impl,
        nb::arg("finite_dgm_a"),
        nb::arg("finite_dgm_b"),
        nb::arg("wasserstein_q") = 1.0,
        nb::arg("wasserstein_delta") = 0.01,
        nb::arg("internal_p") = hera::get_infinity<oin_real>(),
        nb::call_guard<nb::gil_scoped_release>(),
        "Compute Wasserstein matching between two finite diagrams (no essential points). "
        "Returns matching information as a WassersteinMatchingFinite object.");

    // Bind BottleneckLongestEdge struct
    nb::class_<BottleneckLongestEdge>(m, "BottleneckLongestEdge",
            "One edge in the bottleneck matching whose length equals the bottleneck distance. "
            "Either idx_a or idx_b may be -1, indicating a diagonal-projected endpoint.")
        .def_ro("idx_a", &BottleneckLongestEdge::idx_a,
            "Index of the endpoint in finite_dgm_a, or -1 if the endpoint is a diagonal projection.")
        .def_ro("idx_b", &BottleneckLongestEdge::idx_b,
            "Index of the endpoint in finite_dgm_b, or -1 if the endpoint is a diagonal projection.")
        .def_ro("length", &BottleneckLongestEdge::length,
            "Length of this edge (equals bottleneck distance).")
        .def_ro("a_x", &BottleneckLongestEdge::a_x)
        .def_ro("a_y", &BottleneckLongestEdge::a_y)
        .def_ro("b_x", &BottleneckLongestEdge::b_x)
        .def_ro("b_y", &BottleneckLongestEdge::b_y);

    // Bind BottleneckMatchingFinite struct
    nb::class_<BottleneckMatchingFinite>(m, "BottleneckMatchingFinite",
            "Full bottleneck matching between two finite diagrams, plus all edges tied for the bottleneck distance.")
        .def_ro("a_to_b", &BottleneckMatchingFinite::a_to_b,
            "Indices in finite_dgm_a matched to finite points in finite_dgm_b.")
        .def_ro("b_from_a", &BottleneckMatchingFinite::b_from_a,
            "Corresponding indices in finite_dgm_b (parallel to a_to_b).")
        .def_ro("a_to_diag", &BottleneckMatchingFinite::a_to_diag,
            "Indices in finite_dgm_a matched to the diagonal.")
        .def_ro("b_to_diag", &BottleneckMatchingFinite::b_to_diag,
            "Indices in finite_dgm_b matched to the diagonal.")
        .def_ro("longest_edges", &BottleneckMatchingFinite::longest_edges,
            "All edges whose length equals `distance` (ties preserved).")
        .def_ro("distance", &BottleneckMatchingFinite::distance,
            "Bottleneck distance (L_infinity) between the finite parts.");

    // Bind bottleneck_matching_finite function (no internal_p: Hera bottleneck is L_infinity-only).
    func_name = "bottleneck_matching_finite";
    m.def(func_name.c_str(),
        &bottleneck_matching_finite_impl,
        nb::arg("finite_dgm_a"),
        nb::arg("finite_dgm_b"),
        nb::arg("delta") = 0.01,
        nb::call_guard<nb::gil_scoped_release>(),
        "Compute bottleneck matching between two finite diagrams (no essential points). "
        "delta=0.0 runs the exact algorithm. Returns a BottleneckMatchingFinite with the full "
        "matching and all edges tied for the bottleneck distance.");

    func_name = "init_frechet_mean_first_diagram";
    m.def(func_name.c_str(),
            [](const nb::list& diagrams) {
                auto diagram_vec = python_object_to_diagrams(diagrams);
                return diagram_to_numpy(oin::init_frechet_mean_first_diagram<oin_real>(diagram_vec));
            },
            nb::arg("diagrams"),
            "Return the first input diagram as a Fréchet-mean initializer.");

    func_name = "init_frechet_mean_random_diagram";
    m.def(func_name.c_str(),
            [](const nb::list& diagrams,
               oin::DiagramPlaneDomain domain,
               oin_real random_noise_scale,
               size_t random_seed) {
                auto diagram_vec = python_object_to_diagrams(diagrams);
                oin::FrechetMeanInitRandomParams<oin_real> params;
                params.domain = domain;
                params.noise_scale = random_noise_scale;
                params.random_seed = random_seed;
                return diagram_to_numpy(oin::init_frechet_mean_random_diagram<oin_real>(diagram_vec, params));
            },
            nb::arg("diagrams"),
            nb::arg("domain") = oin::DiagramPlaneDomain::AboveDiagonal,
            nb::arg("random_noise_scale") = 1.0,
            nb::arg("random_seed") = 42,
            "Return a random perturbed input diagram as a Fréchet-mean initializer.");

    func_name = "init_frechet_mean_medoid_diagram";
    m.def(func_name.c_str(),
            [](const nb::list& diagrams,
               nb::object weights) {
                auto diagram_vec = python_object_to_diagrams(diagrams);
                std::vector<oin_real> weight_vec;
                if (!weights.is_none()) {
                    nb::sequence weight_seq = nb::cast<nb::sequence>(weights);
                    weight_vec.reserve(nb::len(weight_seq));
                    for (auto item : weight_seq)
                        weight_vec.push_back(nb::cast<oin_real>(item));
                }
                return diagram_to_numpy(oin::init_frechet_mean_medoid_diagram<oin_real>(diagram_vec, weight_vec));
            },
            nb::arg("diagrams"),
            nb::arg("weights") = nb::none(),
            "Return the weighted Wasserstein medoid diagram used as a Fréchet-mean initializer.");

    func_name = "init_frechet_mean_diagonal_grid";
    m.def(func_name.c_str(),
            [](const nb::list& diagrams,
               nb::object weights,
               oin::DiagramPlaneDomain domain,
               size_t grid_n_x_bins,
               size_t grid_n_y_bins) {
                auto diagram_vec = python_object_to_diagrams(diagrams);
                std::vector<oin_real> weight_vec;
                if (!weights.is_none()) {
                    nb::sequence weight_seq = nb::cast<nb::sequence>(weights);
                    weight_vec.reserve(nb::len(weight_seq));
                    for (auto item : weight_seq)
                        weight_vec.push_back(nb::cast<oin_real>(item));
                }
                oin::FrechetMeanInitGridParams params;
                params.domain = domain;
                params.n_x_bins = std::max<size_t>(1, grid_n_x_bins);
                params.n_y_bins = std::max<size_t>(1, grid_n_y_bins);
                return diagram_to_numpy(oin::init_frechet_mean_diagonal_grid<oin_real>(diagram_vec, weight_vec, params));
            },
            nb::arg("diagrams"),
            nb::arg("weights") = nb::none(),
            nb::arg("domain") = oin::DiagramPlaneDomain::AboveDiagonal,
            nb::arg("grid_n_x_bins") = 16,
            nb::arg("grid_n_y_bins") = 16,
            "Return the diagonal-grid Fréchet-mean initializer.");

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
