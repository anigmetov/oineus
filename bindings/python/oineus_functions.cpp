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

// Allocate a (n, 2) int64 ndarray from a vector of (int, int) pairs.
// One linear copy; the user has explicitly opted in to plain copies for
// matching index transport.
nb::ndarray<int64_t, nb::numpy> pairs_to_numpy(const std::vector<std::pair<int, int>>& pairs)
{
    size_t n = pairs.size();
    auto* ptr = new int64_t[n * 2];
    for (size_t i = 0; i < n; ++i) {
        ptr[2 * i]     = pairs[i].first;
        ptr[2 * i + 1] = pairs[i].second;
    }
    nb::capsule owner(ptr, [](void* p) noexcept {
        delete[] static_cast<int64_t*>(p);
    });
    return nb::ndarray<int64_t, nb::numpy>(
        ptr, {n, static_cast<size_t>(2)}, owner);
}

// Allocate a (n,) int64 ndarray from a vector<int>.
nb::ndarray<int64_t, nb::numpy> ints_to_numpy(const std::vector<int>& ints)
{
    size_t n = ints.size();
    auto* ptr = new int64_t[n];
    for (size_t i = 0; i < n; ++i) ptr[i] = ints[i];
    nb::capsule owner(ptr, [](void* p) noexcept {
        delete[] static_cast<int64_t*>(p);
    });
    return nb::ndarray<int64_t, nb::numpy>(ptr, {n}, owner);
}

// Strip essential (infinite-coordinate) points from a diagram in place.
// Used when ignore_inf_points=True.
void strip_essentials(Diagram& dgm)
{
    dgm.erase(std::remove_if(dgm.begin(), dgm.end(),
        [](const DiagramPoint& p) {
            return !std::isfinite(p.birth) || !std::isfinite(p.death);
        }), dgm.end());
}

// ---------------------------------------------------------------------------
// Grouped views over the per-family essential pair / longest-edge arrays.
//
// These are presentation-only adapters that expose the four families
// (inf_death, neg_inf_death, inf_birth, neg_inf_birth) as both attribute
// access (m.essential.inf_death) and dict-like indexing (m.essential["..."]
// or m.essential[InfKind.INF_DEATH]). They hold a const pointer to the
// parent Hera struct's array; nanobind keep_alive on the parent's
// `essential`/`longest` properties keeps the parent live for the view's
// lifetime.
// ---------------------------------------------------------------------------

constexpr const char* kInfKindNames[hera::kNumInfKinds] = {
    "inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"
};

// Resolve a Python key (str | InfKind) into [0, 4) or throw KeyError.
int parse_inf_kind_key(nb::handle key)
{
    // InfKind enum case
    if (nb::isinstance<hera::InfKind>(key)) {
        return static_cast<int>(nb::cast<hera::InfKind>(key));
    }
    // String case
    if (nb::isinstance<nb::str>(key)) {
        std::string s = nb::cast<std::string>(key);
        for (int k = 0; k < hera::kNumInfKinds; ++k)
            if (s == kInfKindNames[k]) return k;
    }
    throw nb::key_error(("Unknown essential family key: " +
                         nb::cast<std::string>(nb::repr(key))).c_str());
}

template<class Real>
struct EssentialMatchesView {
    using Inner = std::array<std::vector<std::pair<int, int>>, hera::kNumInfKinds>;
    const Inner* arrays;
    explicit EssentialMatchesView(const Inner& a) : arrays(&a) {}
    nb::ndarray<int64_t, nb::numpy> get(int k) const { return pairs_to_numpy((*arrays)[k]); }
};

template<class Real>
struct EssentialLongestEdgesView {
    using Inner = std::array<std::vector<hera::EssentialLongestEdge<Real>>, hera::kNumInfKinds>;
    const Inner* arrays;
    explicit EssentialLongestEdgesView(const Inner& a) : arrays(&a) {}
    const std::vector<hera::EssentialLongestEdge<Real>>& get(int k) const { return (*arrays)[k]; }
};

template<class Real>
struct LongestEdgesView {
    const std::vector<hera::FiniteLongestEdge<Real>>* finite;
    EssentialLongestEdgesView<Real> essential;
    LongestEdgesView(const std::vector<hera::FiniteLongestEdge<Real>>& f,
                     const typename EssentialLongestEdgesView<Real>::Inner& e)
        : finite(&f), essential(e) {}
};


std::vector<Diagram> python_object_to_diagrams(const nb::list& diagrams)
{
    std::vector<Diagram> result;
    result.reserve(nb::len(diagrams));

    for (auto item : diagrams)
        result.push_back(python_object_to_diagram(item));

    return result;
}

// Build a hera::Diagram from a numpy (n, 2) array, assigning each point
// id = its original position in the input. This is the entry point to the
// new detailed Hera matching functions; the user-side `id` flows through
// auction + 1D essential matching and surfaces in the returned struct.
Diagram numpy_to_diagram_with_pos_ids(const NumpyDiagram& dgm)
{
    if (dgm.ndim() != 2 || dgm.shape(1) != 2)
        throw nb::value_error("Expected NumPy array with shape (n_points, 2)");
    Diagram out;
    out.reserve(dgm.shape(0));
    const auto* p = static_cast<const oin_real*>(dgm.data());
    for (size_t i = 0; i < dgm.shape(0); ++i) {
        out.emplace_back(p[2 * i], p[2 * i + 1]);
        out.back().id = static_cast<oin::id_type>(i);
    }
    return out;
}

hera::WassersteinMatching<oin_real> wasserstein_matching_detailed_impl(
    const NumpyDiagram& dgm_a_np,
    const NumpyDiagram& dgm_b_np,
    oin_real q,
    oin_real delta,
    oin_real internal_p,
    bool ignore_inf_points)
{
    Diagram dgm_a = numpy_to_diagram_with_pos_ids(dgm_a_np);
    Diagram dgm_b = numpy_to_diagram_with_pos_ids(dgm_b_np);

    if (ignore_inf_points) {
        strip_essentials(dgm_a);
        strip_essentials(dgm_b);
    }

    hera::AuctionParams<oin_real> params;
    params.wasserstein_power = q;
    params.delta             = delta;
    params.internal_p        = internal_p;

    return hera::wasserstein_matching_detailed(dgm_a, dgm_b, params);
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

    // In-order (VRE) Vietoris-Rips construction.
    func_name = "get_vr_filtration_inorder";
    m.def(func_name.c_str(), &get_vr_filtration_inorder<oin_int, oin_real>,
            nb::arg("points"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    func_name = "get_vr_filtration_and_critical_edges_inorder";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges_inorder<oin_int, oin_real>,
            nb::arg("points"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    func_name = "get_vr_filtration_inorder_from_pwdists";
    m.def(func_name.c_str(), &get_vr_filtration_inorder_from_pwdists<oin_int, oin_real>,
            nb::arg("pwdists"), nb::arg("max_dim"), nb::arg("max_diameter")=std::numeric_limits<oin_real>::max(), nb::arg("n_threads")=1);

    func_name = "get_vr_filtration_and_critical_edges_inorder_from_pwdists";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges_inorder_from_pwdists<oin_int, oin_real>,
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

    // ---- enum + grouped views ----
    nb::enum_<hera::InfKind>(m, "InfKind",
            "The four families of essential (infinite-coordinate) diagram points.")
        .value("INF_DEATH",      hera::InfKind::InfDeath,
               "(finite, +inf) — homology class born finitely, never dies.")
        .value("NEG_INF_DEATH",  hera::InfKind::NegInfDeath,
               "(finite, -inf).")
        .value("INF_BIRTH",      hera::InfKind::InfBirth,
               "(+inf, finite).")
        .value("NEG_INF_BIRTH",  hera::InfKind::NegInfBirth,
               "(-inf, finite).");

    using EssView = EssentialMatchesView<oin_real>;
    nb::class_<EssView>(m, "EssentialMatches",
            "Grouped view over essential-point matches by family.")
        .def_prop_ro("inf_death",
            [](const EssView& v) { return v.get(0); }, nb::rv_policy::move)
        .def_prop_ro("neg_inf_death",
            [](const EssView& v) { return v.get(1); }, nb::rv_policy::move)
        .def_prop_ro("inf_birth",
            [](const EssView& v) { return v.get(2); }, nb::rv_policy::move)
        .def_prop_ro("neg_inf_birth",
            [](const EssView& v) { return v.get(3); }, nb::rv_policy::move)
        .def("__getitem__",
            [](const EssView& v, nb::handle key) { return v.get(parse_inf_kind_key(key)); },
            nb::rv_policy::move)
        .def("__contains__",
            [](const EssView&, nb::handle key) {
                try { parse_inf_kind_key(key); return true; }
                catch (...) { return false; }
            })
        .def("__iter__",
            [](const EssView&) {
                return nb::iter(nb::cast(std::vector<std::string>{
                    kInfKindNames[0], kInfKindNames[1],
                    kInfKindNames[2], kInfKindNames[3]}));
            })
        .def("__len__", [](const EssView&) { return hera::kNumInfKinds; })
        .def("keys",
            [](const EssView&) {
                return nb::cast(std::vector<std::string>{
                    kInfKindNames[0], kInfKindNames[1],
                    kInfKindNames[2], kInfKindNames[3]});
            })
        .def("values",
            [](const EssView& v) {
                nb::list out;
                for (int k = 0; k < hera::kNumInfKinds; ++k) out.append(v.get(k));
                return out;
            })
        .def("items",
            [](const EssView& v) {
                nb::list out;
                for (int k = 0; k < hera::kNumInfKinds; ++k) {
                    nb::tuple t = nb::make_tuple(kInfKindNames[k], v.get(k));
                    out.append(t);
                }
                return out;
            })
        .def("__repr__",
            [](const EssView& v) {
                std::stringstream ss;
                ss << "EssentialMatches(";
                for (int k = 0; k < hera::kNumInfKinds; ++k) {
                    if (k) ss << ", ";
                    ss << kInfKindNames[k] << "=" << (*v.arrays)[k].size();
                }
                ss << ")";
                return ss.str();
            });

    using EssLongView = EssentialLongestEdgesView<oin_real>;
    nb::class_<EssLongView>(m, "EssentialLongestEdges",
            "Grouped view over per-family tied-longest edges.")
        .def_prop_ro("inf_death",
            [](const EssLongView& v) { return v.get(0); })
        .def_prop_ro("neg_inf_death",
            [](const EssLongView& v) { return v.get(1); })
        .def_prop_ro("inf_birth",
            [](const EssLongView& v) { return v.get(2); })
        .def_prop_ro("neg_inf_birth",
            [](const EssLongView& v) { return v.get(3); })
        .def("__getitem__",
            [](const EssLongView& v, nb::handle key) {
                return v.get(parse_inf_kind_key(key));
            })
        .def("__contains__",
            [](const EssLongView&, nb::handle key) {
                try { parse_inf_kind_key(key); return true; }
                catch (...) { return false; }
            })
        .def("__iter__",
            [](const EssLongView&) {
                return nb::iter(nb::cast(std::vector<std::string>{
                    kInfKindNames[0], kInfKindNames[1],
                    kInfKindNames[2], kInfKindNames[3]}));
            })
        .def("__len__", [](const EssLongView&) { return hera::kNumInfKinds; })
        .def("keys",
            [](const EssLongView&) {
                return nb::cast(std::vector<std::string>{
                    kInfKindNames[0], kInfKindNames[1],
                    kInfKindNames[2], kInfKindNames[3]});
            })
        .def("values",
            [](const EssLongView& v) {
                nb::list out;
                for (int k = 0; k < hera::kNumInfKinds; ++k) out.append(v.get(k));
                return out;
            })
        .def("items",
            [](const EssLongView& v) {
                nb::list out;
                for (int k = 0; k < hera::kNumInfKinds; ++k)
                    out.append(nb::make_tuple(kInfKindNames[k], v.get(k)));
                return out;
            })
        .def("__repr__",
            [](const EssLongView& v) {
                std::stringstream ss;
                ss << "EssentialLongestEdges(";
                for (int k = 0; k < hera::kNumInfKinds; ++k) {
                    if (k) ss << ", ";
                    ss << kInfKindNames[k] << "=" << (*v.arrays)[k].size();
                }
                ss << ")";
                return ss.str();
            });

    using LongView = LongestEdgesView<oin_real>;
    nb::class_<LongView>(m, "LongestEdges",
            "Bottleneck longest-edge data, split into finite and essential parts.")
        .def_prop_ro("finite",
            [](const LongView& v) { return *v.finite; })
        .def_prop_ro("essential",
            [](LongView& v) -> EssLongView& { return v.essential; },
            nb::rv_policy::reference_internal)
        .def("__repr__",
            [](const LongView& v) {
                int ess_total = 0;
                for (int k = 0; k < hera::kNumInfKinds; ++k)
                    ess_total += (*v.essential.arrays)[k].size();
                std::stringstream ss;
                ss << "LongestEdges(finite=" << v.finite->size()
                   << ", essential=" << ess_total << ")";
                return ss.str();
            });


    // ---- DiagramMatching: hera::WassersteinMatching, exposed under the
    // user-facing Python name. Keeps the grouped-view interface.  ----
    using HeraWasserMatch = hera::WassersteinMatching<oin_real>;
    nb::class_<HeraWasserMatch>(m, "DiagramMatching",
            "Optimal Wasserstein matching: bucketed pair indices, cost, distance.")
        .def_prop_ro("finite_to_finite",
            [](const HeraWasserMatch& self) { return pairs_to_numpy(self.finite_to_finite); },
            nb::rv_policy::move,
            "(n, 2) int64 ndarray of (idx_a, idx_b) pairs for finite-to-finite matches.")
        .def_prop_ro("a_to_diagonal",
            [](const HeraWasserMatch& self) { return ints_to_numpy(self.a_to_diagonal); },
            nb::rv_policy::move,
            "1-D int64 ndarray of indices in dgm_a matched to the diagonal.")
        .def_prop_ro("b_to_diagonal",
            [](const HeraWasserMatch& self) { return ints_to_numpy(self.b_to_diagonal); },
            nb::rv_policy::move,
            "1-D int64 ndarray of indices in dgm_b matched to the diagonal.")
        .def_prop_ro("essential",
            [](const HeraWasserMatch& self) {
                return EssentialMatchesView<oin_real>(self.essential);
            },
            nb::keep_alive<0, 1>(),
            "Grouped view of essential-point matches per family.")
        .def_ro("cost", &HeraWasserMatch::cost,
            "Total Wasserstein cost (== distance ** q).")
        .def_ro("distance", &HeraWasserMatch::distance,
            "Wasserstein distance.")
        .def("__str__",
            [](const HeraWasserMatch& self) {
                // Substitute the C++ name with the Python class name for the
                // user-facing summary.
                std::stringstream ss; ss << self;
                std::string s = ss.str();
                if (s.compare(0, 19, "WassersteinMatching") == 0)
                    s.replace(0, 19, "DiagramMatching");
                return s;
            })
        .def("__repr__",
            [](const HeraWasserMatch& self) {
                std::string s = hera::to_str_debug(self);
                if (s.compare(0, 19, "WassersteinMatching") == 0)
                    s.replace(0, 19, "DiagramMatching");
                return s;
            });

    func_name = "wasserstein_matching_detailed";
    m.def(func_name.c_str(),
        &wasserstein_matching_detailed_impl,
        nb::arg("dgm_a"),
        nb::arg("dgm_b"),
        nb::arg("wasserstein_q") = 2.0,
        nb::arg("wasserstein_delta") = 0.01,
        nb::arg("internal_p") = hera::get_infinity<oin_real>(),
        nb::arg("ignore_inf_points") = true,
        nb::call_guard<nb::gil_scoped_release>(),
        "Full Wasserstein matching: cost, distance, and bucketed pair indices "
        "for finite-to-finite, a-to-diagonal, b-to-diagonal, and the four "
        "essential families. ignore_inf_points=True drops essentials before matching.");


    // ---- bottleneck longest-edge records + matching ----
    using HeraBtFiniteEdge = hera::FiniteLongestEdge<oin_real>;
    nb::class_<HeraBtFiniteEdge>(m, "FiniteLongestEdge",
            "One edge tied for the bottleneck distance in the finite part of "
            "a matching. ``idx_a`` or ``idx_b`` is ``None`` if that endpoint "
            "is a diagonal projection.")
        .def_ro("length", &HeraBtFiniteEdge::length)
        .def_prop_ro("idx_a",
            [](const HeraBtFiniteEdge& e) -> nb::object {
                return e.idx_a < 0 ? nb::none() : nb::cast(e.idx_a);
            })
        .def_prop_ro("idx_b",
            [](const HeraBtFiniteEdge& e) -> nb::object {
                return e.idx_b < 0 ? nb::none() : nb::cast(e.idx_b);
            })
        .def_prop_ro("point_a",
            [](const HeraBtFiniteEdge& e) {
                return nb::make_tuple(e.a_x, e.a_y);
            })
        .def_prop_ro("point_b",
            [](const HeraBtFiniteEdge& e) {
                return nb::make_tuple(e.b_x, e.b_y);
            })
        .def("__repr__",
            [](const HeraBtFiniteEdge& e) {
                std::stringstream ss;
                auto fmt_idx = [](int i) -> std::string {
                    return i < 0 ? std::string("None") : std::to_string(i);
                };
                ss << "FiniteLongestEdge(length=" << e.length
                   << ", idx_a=" << fmt_idx(e.idx_a)
                   << ", idx_b=" << fmt_idx(e.idx_b)
                   << ", point_a=(" << e.a_x << ", " << e.a_y << ")"
                   << ", point_b=(" << e.b_x << ", " << e.b_y << "))";
                return ss.str();
            });

    using HeraBtEssEdge = hera::EssentialLongestEdge<oin_real>;
    nb::class_<HeraBtEssEdge>(m, "EssentialLongestEdge",
            "One edge tied for the bottleneck distance within a single "
            "essential family.")
        .def_ro("length",  &HeraBtEssEdge::length)
        .def_ro("idx_a",   &HeraBtEssEdge::idx_a)
        .def_ro("idx_b",   &HeraBtEssEdge::idx_b)
        .def_ro("coord_a", &HeraBtEssEdge::coord_a)
        .def_ro("coord_b", &HeraBtEssEdge::coord_b)
        .def("__repr__",
            [](const HeraBtEssEdge& e) {
                std::stringstream ss;
                ss << "EssentialLongestEdge(length=" << e.length
                   << ", idx_a=" << e.idx_a << ", idx_b=" << e.idx_b
                   << ", coord_a=" << e.coord_a << ", coord_b=" << e.coord_b << ")";
                return ss.str();
            });

    using HeraBtMatch = hera::BottleneckMatching<oin_real>;
    // Inherits all properties of DiagramMatching (finite_to_finite,
    // a_to_diagonal, b_to_diagonal, essential, cost, distance) via the
    // C++ inheritance declared in hera::BottleneckMatching.
    nb::class_<HeraBtMatch, HeraWasserMatch>(m, "BottleneckMatching",
            "Optimal bottleneck matching: inherits the DiagramMatching "
            "interface and adds longest-edge data via the .longest view.")
        .def_prop_ro("longest",
            [](const HeraBtMatch& s) {
                return LongestEdgesView<oin_real>(s.longest_finite, s.longest_essential);
            },
            nb::keep_alive<0, 1>(),
            "View over tied-longest edges, with `.finite` (a list of "
            "FiniteLongestEdge) and `.essential` (an EssentialLongestEdges "
            "grouped view).")
        .def("__str__",
            [](const HeraBtMatch& s) { std::stringstream ss; ss << s; return ss.str(); })
        .def("__repr__",
            [](const HeraBtMatch& s) { return hera::to_str_debug(s); });

    func_name = "bottleneck_matching_detailed";
    m.def(func_name.c_str(),
        [](const NumpyDiagram& dgm_a_np, const NumpyDiagram& dgm_b_np,
           oin_real delta, bool ignore_inf_points) {
            Diagram dgm_a = numpy_to_diagram_with_pos_ids(dgm_a_np);
            Diagram dgm_b = numpy_to_diagram_with_pos_ids(dgm_b_np);
            if (ignore_inf_points) {
                strip_essentials(dgm_a);
                strip_essentials(dgm_b);
            }
            return hera::bottleneck_matching_detailed(dgm_a, dgm_b, delta);
        },
        nb::arg("dgm_a"),
        nb::arg("dgm_b"),
        nb::arg("delta") = 0.01,
        nb::arg("ignore_inf_points") = true,
        nb::call_guard<nb::gil_scoped_release>(),
        "Full bottleneck matching: distance, bucketed pair indices for "
        "finite-to-finite/diagonal/essentials, and tied-longest edges. "
        "delta=0.0 runs the exact algorithm.");

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
               nb::object weights,
               int n_threads) {
                auto diagram_vec = python_object_to_diagrams(diagrams);
                std::vector<oin_real> weight_vec;
                if (!weights.is_none()) {
                    nb::sequence weight_seq = nb::cast<nb::sequence>(weights);
                    weight_vec.reserve(nb::len(weight_seq));
                    for (auto item : weight_seq)
                        weight_vec.push_back(nb::cast<oin_real>(item));
                }
                typename oin::Diagrams<oin_real>::Dgm result;
                {
                    nb::gil_scoped_release release;
                    result = oin::init_frechet_mean_medoid_diagram<oin_real>(
                            diagram_vec, weight_vec, std::max(1, n_threads));
                }
                return diagram_to_numpy(result);
            },
            nb::arg("diagrams"),
            nb::arg("weights") = nb::none(),
            nb::arg("n_threads") = 1,
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
               nb::object custom_initial_barycenter,
               int n_threads) {
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
                params.n_threads = std::max(1, n_threads);
                params.random_init_params.noise_scale = random_noise_scale;
                params.random_init_params.random_seed = random_seed;
                params.random_init_params.domain = domain;
                params.grid_init_params.n_x_bins = std::max<size_t>(1, grid_n_x_bins);
                params.grid_init_params.n_y_bins = std::max<size_t>(1, grid_n_y_bins);
                params.grid_init_params.domain = domain;

                Diagram custom_dgm;
                if (!custom_initial_barycenter.is_none())
                    custom_dgm = python_object_to_diagram(custom_initial_barycenter);

                typename oin::Diagrams<oin_real>::Dgm result;
                {
                    nb::gil_scoped_release release;
                    result = oin::frechet_mean<oin_real>(diagram_vec, weight_vec, params, custom_dgm);
                }
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
            nb::arg("n_threads") = 1,
            "Compute a Frechet mean (W2 barycenter) of persistence diagrams.");
}
