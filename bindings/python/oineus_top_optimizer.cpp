#include "oineus_persistence_bindings.h"
#include "oineus_type_list.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/map.h"
#include "nanobind/stl/unordered_map.h"

    //
    // std::pair<IndicesValues, Real> match_and_distance(typename Diagrams<Real>::Dgm& template_dgm, dim_type d, Real wasserstein_q)
    // {
    //     // set ids in template diagram
    //     for(size_t i = 0 ; i < template_dgm.size() ; ++i) {
    //         template_dgm[i].id = i;
    //
    //         if (template_dgm[i].is_inf())
    //             throw std::runtime_error("infinite point in template diagram");
    //     }
    //
    //     using Diagram = typename Diagrams<Real>::Dgm;
    //
    //     IndicesValues result;
    //
    //     hera::AuctionParams<Real> hera_params;
    //     hera_params.return_matching = true;
    //     hera_params.match_inf_points = false;
    //     hera_params.wasserstein_power = wasserstein_q;
    //
    //     if (not decmp_hom_.is_reduced)
    //         decmp_hom_.reduce(params_hom_);
    //
    //     Diagram current_dgm = decmp_hom_.diagram(fil_, false).get_diagram_in_dimension(d);
    //
    //     for(size_t i = 0 ; i < current_dgm.size() ; ++i) {
    //         current_dgm[i].id = i;
    //     }
    //
    //     // template_dgm: bidders, a
    //     // current_dgm: items, b
    //     auto hera_res = hera::wasserstein_cost_detailed<Diagram>(template_dgm, current_dgm, hera_params);
    //
    //     for(auto curr_template: hera_res.matching_b_to_a_) {
    //         auto current_id = curr_template.first;
    //         auto template_id = curr_template.second;
    //
    //         if (current_id < 0)
    //             continue;
    //
    //         size_t birth_idx = current_dgm.at(current_id).birth_index;
    //         size_t death_idx = current_dgm.at(current_id).death_index;
    //
    //         Real birth_target;
    //         Real death_target;
    //
    //         if (template_id >= 0) {
    //             // matched to off-diagonal point of template diagram
    //
    //             birth_target = template_dgm.at(template_id).birth;
    //             death_target = template_dgm.at(template_id).death;
    //         } else {
    //             // matched to diagonal point of template diagram
    //             auto curr_proj_id = -template_id - 1;
    //             Real m = (current_dgm.at(curr_proj_id).birth + current_dgm.at(curr_proj_id).death) / 2;
    //             birth_target = death_target = m;
    //         }
    //
    //         result.push_back(birth_idx, birth_target);
    //         result.push_back(death_idx, death_target);
    //     }
    //
    //     return {result, hera_res.distance};
    // }


template<class Cell>
void init_oineus_top_optimizer_class(nb::module_& m, std::string opt_name, std::string ind_vals_name)
{
    using Filtration = oin::Filtration<Cell, oin_real>;
    using TopologyOptimizer = oin::TopologyOptimizer<Cell, oin_real>;
    using IndicesValues = typename TopologyOptimizer::IndicesValues;
    using CriticalSets = typename TopologyOptimizer::CriticalSets;
    using ConflictStrategy = oin::ConflictStrategy;
    using Diagram = typename oin::Diagrams<oin_real>::Dgm;
    using Indices = typename TopologyOptimizer::Indices;
    using Values = typename TopologyOptimizer::Values;
    using IndicesValues = typename TopologyOptimizer::IndicesValues;
    using Indices = typename TopologyOptimizer::Indices;
    using Values = typename TopologyOptimizer::Values;
    using IndicesValuesStateTuple = std::tuple<Indices, Values>;
    using TopologyOptimizerStateTuple = std::tuple<decltype(TopologyOptimizer::negate_),
                                                   decltype(TopologyOptimizer::fil_),
                                                   decltype(TopologyOptimizer::decmp_hom_),
                                                   decltype(TopologyOptimizer::decmp_coh_),
                                                   decltype(TopologyOptimizer::params_hom_),
                                                   decltype(TopologyOptimizer::params_coh_),
                                                   decltype(TopologyOptimizer::with_crit_sets_),
                                                   decltype(TopologyOptimizer::n_threads_),
                                                   decltype(TopologyOptimizer::dims_to_restore_elz_),
                                                   decltype(TopologyOptimizer::u_strategy_),
                                                   decltype(TopologyOptimizer::decmp_hom_built_),
                                                   decltype(TopologyOptimizer::decmp_coh_built_)>;

    nb::class_<IndicesValues>(m, ind_vals_name.c_str())
            .def("__getitem__", [](const IndicesValues& iv, int i) -> std::variant<Indices, Values> {
              if (i == 0)
                  return {iv.indices};
              else if (i == 1)
                  return {iv.values};
              else
                  throw std::out_of_range("IndicesValues: i must be  0 or 1");
            })
            // Zero-copy numpy views; lifetime is tied to the IndicesValues
            // instance via nb::rv_policy::reference_internal.
            .def("indices_array",
                 [](IndicesValues& iv) {
                   return nb::ndarray<typename Indices::value_type, nb::numpy>(
                       iv.indices.data(),
                       {iv.indices.size()},
                       nb::find(&iv));
                 },
                 nb::rv_policy::reference_internal,
                 "Zero-copy numpy view over the indices vector. The view "
                 "is invalidated by any operation that mutates the underlying "
                 "IndicesValues; copy via np.array(...) to detach.")
            .def("values_array",
                 [](IndicesValues& iv) {
                   return nb::ndarray<oin_real, nb::numpy>(
                       iv.values.data(),
                       {iv.values.size()},
                       nb::find(&iv));
                 },
                 nb::rv_policy::reference_internal,
                 "Zero-copy numpy view over the values vector.")
            .def("__repr__", [](const IndicesValues& iv) {
              std::stringstream ss;
              ss << iv;
              return ss.str();
            })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const IndicesValues& iv) -> IndicesValuesStateTuple {
                return std::make_tuple(iv.indices, iv.values);
            })
            .def("__setstate__", [](IndicesValues& iv, const IndicesValuesStateTuple& t) {
                new (&iv) IndicesValues();
                iv.indices = std::get<0>(t);
                iv.values = std::get<1>(t);
            });

    // optimization
    nb::class_<TopologyOptimizer>(m, opt_name.c_str())
            .def(nb::init<const Filtration&, bool, oin::DimVec, int, oin::UStrategy>(),
                 nb::arg("fil"),
                 nb::arg("with_crit_sets") = true,
                 nb::arg("dims_to_restore_elz") = oin::DimVec{},
                 nb::arg("n_threads") = 1,
                 nb::arg("u_strategy") = oin::UStrategy::Auto,
                 "Construct a TopologyOptimizer for one autograd backward. "
                 "with_crit_sets=false sets up the cheapest reduction (R only, "
                 "no V, no U) for diagram-loss; with_crit_sets=true sets up V "
                 "(plus restore_ELZ in dims_to_restore_elz) so that "
                 "ensure_has_u_* can recover U on demand. u_strategy picks "
                 "which equation to solve for U; LegacyInBand forces a "
                 "serial in-band U during the reduction.")
            .def("compute_diagram", [](TopologyOptimizer& opt, bool include_inf_points) { return PyOineusDiagrams<oin_real>(opt.compute_diagram(include_inf_points)); },
                    nb::arg("include_inf_points"),
                    "compute diagrams in all dimensions")
            .def("simplify", &TopologyOptimizer::simplify,
                    nb::arg("epsilon"),
                    nb::arg("strategy"),
                    nb::arg("dim"), "make points with persistence less than epsilon go to the diagonal")
            .def("get_nth_persistence", &TopologyOptimizer::get_nth_persistence,
                    nb::arg("dim"), nb::arg("n"), "return n-th persistence value in d-dimensional persistence diagram")
            .def("match", [](TopologyOptimizer& opt, Diagram& template_dgm, dim_type d, oin_real wasserstein_q, oin_real wasserstein_delta, bool return_wasserstein_distance, bool dualize) ->
                std::variant<IndicesValues, std::pair<IndicesValues, oin_real>>
                    {
                      if (return_wasserstein_distance)
                          return opt.match_and_distance(template_dgm, d, wasserstein_q, wasserstein_delta, dualize);
                      else
                          return opt.match(template_dgm, d, wasserstein_q, wasserstein_delta, dualize);
                    },
                    nb::arg("template_dgm"),
                    nb::arg("dim"),
                    nb::arg("q")=1.0,
                    nb::arg("wasserstein_delta")=0.01,
                    nb::arg("return_wasserstein_distance")=false,
                    nb::arg("dualize")=false,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "return target from Wasserstein matching"
            )
            .def_prop_ro("homology_decomposition", &TopologyOptimizer::get_homology_decompostion)
            .def_prop_ro("cohomology_decomposition", &TopologyOptimizer::get_cohomology_decompostion)
            // Mutable references to the internal decompositions, used by
            // the partial-U wiring in oineus.diff.persistence_diagram (so
            // it can call decmp.compute_partial_u_from_v_1 or
            // decmp.compute_partial_u_rows directly on the live optimizer
            // state). The non-_ref names above return copies and are kept
            // for API compatibility.
            .def("homology_decomposition_ref",
                 [](TopologyOptimizer& opt) -> typename TopologyOptimizer::Decomposition& {
                     if (not opt.decmp_hom_built_)
                         throw std::runtime_error("homology_decomposition_ref unavailable; call ensure_hom_built() or ensure_hom_reduced() first");
                     return opt.decmp_hom_;
                 },
                 nb::rv_policy::reference_internal)
            .def("cohomology_decomposition_ref",
                 [](TopologyOptimizer& opt) -> typename TopologyOptimizer::Decomposition& {
                     if (not opt.decmp_coh_built_)
                         throw std::runtime_error("cohomology_decomposition_ref unavailable; call ensure_coh_built() or ensure_coh_reduced() first");
                     return opt.decmp_coh_;
                 },
                 nb::rv_policy::reference_internal)
            .def_prop_ro("is_hom_built", &TopologyOptimizer::is_hom_built,
                    "True iff the homology Decomposition has been materialized "
                    "from the cached boundary matrix. Set by the first "
                    "ensure_hom_built / ensure_hom_reduced call.")
            .def_prop_ro("is_coh_built", &TopologyOptimizer::is_coh_built,
                    "True iff the cohomology Decomposition has been "
                    "materialized. Counterpart of is_hom_built.")
            .def("singleton", &TopologyOptimizer::singleton)
            .def("singletons", &TopologyOptimizer::singletons,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("reduce_all", &TopologyOptimizer::reduce_all,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("increase_death", nb::overload_cast<size_t>(&TopologyOptimizer::increase_death, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for increasing death to inf")
            .def("decrease_death", nb::overload_cast<size_t>(&TopologyOptimizer::decrease_death, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for decreasing death to -inf")
            .def("increase_birth", nb::overload_cast<size_t>(&TopologyOptimizer::increase_birth, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for increasing birth to inf")
            .def("decrease_birth", nb::overload_cast<size_t>(&TopologyOptimizer::decrease_birth, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for decreasing birth to -inf")
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, ConflictStrategy)>(&TopologyOptimizer::combine_loss),
                    nb::arg("critical_sets"),
                    nb::arg("strategy"),
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "combine critical sets into well-defined assignment of new values to indices")
            .def("crit_sets_apply", &TopologyOptimizer::crit_sets_apply,
                    nb::arg("indices"), nb::arg("values"), nb::arg("strategy"),
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Fused per-pair critical-set walk + conflict "
                    "resolution in one C++ call. Internally calls "
                    "ensure_hom_reduced (for the per-pair is_negative "
                    "dispatch) and ensure_has_u_* on the relevant side "
                    "when U is needed. Returns IndicesValues; use "
                    ".indices_array() / .values_array() for zero-copy "
                    "numpy.")
            .def("ensure_hom_built", &TopologyOptimizer::ensure_hom_built,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Materialize the homology Decomposition from the cached "
                    "boundary matrix (no reduction). Idempotent. Use this when "
                    "you need access to the decomposition object but not yet "
                    "to a reduced state. Most callers want ensure_hom_reduced.")
            .def("ensure_coh_built", &TopologyOptimizer::ensure_coh_built,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Cohomology counterpart of ensure_hom_built.")
            .def("ensure_hom_reduced", &TopologyOptimizer::ensure_hom_reduced,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Reduce the homology side with the recipe baked in at "
                    "construction time. Builds the decomposition first if "
                    "needed. Idempotent: no-op if already reduced.")
            .def("ensure_coh_reduced", &TopologyOptimizer::ensure_coh_reduced,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Reduce the cohomology side with the recipe baked in at "
                    "construction time. Builds the decomposition first if "
                    "needed. Idempotent.")
            .def("ensure_has_u_hom",
                    &TopologyOptimizer::ensure_has_u_hom,
                    nb::arg("dim"),
                    nb::arg("rows_fil"),
                    nb::arg("bounds"),
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Ensure U-row data is available on the hom side over the "
                    "given filtration row indices (each in geometric `dim`) "
                    "with the matching value bounds. No-op for LegacyInBand "
                    "(U is already built in-band during reduction).")
            .def("ensure_has_u_coh",
                    &TopologyOptimizer::ensure_has_u_coh,
                    nb::arg("dim"),
                    nb::arg("rows_fil"),
                    nb::arg("bounds"),
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Cohomology-side counterpart of ensure_has_u_hom. The "
                    "row indices are passed as filtration indices; conversion "
                    "to matrix layout (fil_size - 1 - i) is done internally.")
            .def("update", &TopologyOptimizer::update)
            .def("__repr__", [](const TopologyOptimizer& opt) {
                std::stringstream ss;
                ss << opt;
                return ss.str();
            })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const TopologyOptimizer& opt) -> TopologyOptimizerStateTuple {
                return std::make_tuple(opt.negate_, opt.fil_, opt.decmp_hom_, opt.decmp_coh_,
                                       opt.params_hom_, opt.params_coh_,
                                       opt.with_crit_sets_, opt.n_threads_,
                                       opt.dims_to_restore_elz_, opt.u_strategy_,
                                       opt.decmp_hom_built_, opt.decmp_coh_built_);
            })
            .def("__setstate__", [](TopologyOptimizer& opt, const TopologyOptimizerStateTuple& t) {
                // Reconstruct in place via the public ctor (rebuilds
                // boundary_data_ from the pickled filtration); then
                // overwrite the per-side fields with the pickled state,
                // including the built_ flags so that subsequent
                // ensure_*_built calls do not clobber the restored decmp.
                new (&opt) TopologyOptimizer(std::get<1>(t), std::get<6>(t),
                                             std::get<8>(t), std::get<7>(t), std::get<9>(t));
                opt.negate_ = std::get<0>(t);
                opt.fil_ = std::get<1>(t);
                opt.decmp_hom_ = std::get<2>(t);
                opt.decmp_coh_ = std::get<3>(t);
                opt.params_hom_ = std::get<4>(t);
                opt.params_coh_ = std::get<5>(t);
                opt.decmp_hom_built_ = std::get<10>(t);
                opt.decmp_coh_built_ = std::get<11>(t);
            });

}

void init_oineus_top_optimizer(nb::module_& m)
{
    using Simp = oin::Simplex<oin_int>;
    using SimpProd = oin::ProductCell<Simp, Simp>;
    using Cube_1D = oin::Cube<oin_int, 1>;
    using Cube_2D = oin::Cube<oin_int, 2>;
    using Cube_3D = oin::Cube<oin_int, 3>;
    using FrCell_1D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 1>>;
    using FrCell_2D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 2>>;
    using FrCell_3D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 3>>;
    using PackedCell_64 = oin::Simplex<oin_int, oin::BitPacked<oin_int, std::uint64_t>>;
    using PackedCell_128 = oin::Simplex<oin_int, oin::BitPacked<oin_int, unsigned __int128>>;

    nb::enum_<oin::UStrategy>(m, "UStrategy",
            "U-computation strategy used by the crit-sets backward in "
            "oineus.diff. Auto resolves to RowPartial; LegacyInBand "
            "is the in-band serial reduction kept as a control.")
            .value("Auto",         oin::UStrategy::Auto)
            .value("RowPartial",   oin::UStrategy::RowPartial)
            .value("LegacyInBand", oin::UStrategy::LegacyInBand);

    init_oineus_top_optimizer_class<Simp>(m, "TopologyOptimizer", "IndicesValues");
    init_oineus_top_optimizer_class<SimpProd>(m, "TopologyOptimizerProd", "IndicesValuesProd");
    init_oineus_top_optimizer_class<Cube_1D>(m, "TopologyOptimizerCube_1D", "IndicesValuesCube_1D");
    init_oineus_top_optimizer_class<Cube_2D>(m, "TopologyOptimizerCube_2D", "IndicesValuesCube_2D");
    init_oineus_top_optimizer_class<Cube_3D>(m, "TopologyOptimizerCube_3D", "IndicesValuesCube_3D");
    init_oineus_top_optimizer_class<FrCell_1D>(m, "TopologyOptimizerFreudenthal_1D", "IndicesValuesFreudenthal_1D");
    init_oineus_top_optimizer_class<FrCell_2D>(m, "TopologyOptimizerFreudenthal_2D", "IndicesValuesFreudenthal_2D");
    init_oineus_top_optimizer_class<FrCell_3D>(m, "TopologyOptimizerFreudenthal_3D", "IndicesValuesFreudenthal_3D");
    init_oineus_top_optimizer_class<PackedCell_64>(m, "TopologyOptimizerPacked_64", "IndicesValuesPacked_64");
    init_oineus_top_optimizer_class<PackedCell_128>(m, "TopologyOptimizerPacked_128", "IndicesValuesPacked_128");

    // induced matching: one binding per cell type, folded (same order as before)
    using OptCellList = oineus_python::TypeList<Simp, SimpProd, Cube_1D, Cube_2D, Cube_3D,
            FrCell_1D, FrCell_2D, FrCell_3D, PackedCell_64, PackedCell_128>;
    oineus_python::for_each_type(OptCellList{}, [&m]<class Cell>() {
        m.def("get_induced_matching", &oin::get_induced_matching<Cell, oin_real>,
                "Compute induced matching for two filtrations of the same complex",
                nb::arg("included_filtration"),
                nb::arg("containing_filtration"),
                nb::arg("dim")=static_cast<dim_type>(-1),
                nb::arg("n_threads")=1);
    });
}
