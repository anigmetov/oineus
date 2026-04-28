#include "oineus_persistence_bindings.h"
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
                                                   decltype(TopologyOptimizer::params_coh_)>;

    nb::class_<IndicesValues>(m, ind_vals_name.c_str())
            .def("__getitem__", [](const IndicesValues& iv, int i) -> std::variant<Indices, Values> {
              if (i == 0)
                  return {iv.indices};
              else if (i == 1)
                  return {iv.values};
              else
                  throw std::out_of_range("IndicesValues: i must be  0 or 1");
            })
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
            .def(nb::init<const Filtration&>())
            .def("compute_diagram", [](TopologyOptimizer& opt, bool include_inf_points) { return PyOineusDiagrams<oin_real>(opt.compute_diagram(include_inf_points)); },
                    nb::arg("include_inf_points"),
                    "compute diagrams in all dimensions")
            .def("simplify", &TopologyOptimizer::simplify,
                    nb::arg("epsilon"),
                    nb::arg("strategy"),
                    nb::arg("dim"), "make points with persistence less than epsilon go to the diagonal")
            .def("get_nth_persistence", &TopologyOptimizer::get_nth_persistence,
                    nb::arg("dim"), nb::arg("n"), "return n-th persistence value in d-dimensional persistence diagram")
            .def("match", [](TopologyOptimizer& opt, Diagram& template_dgm, dim_type d, oin_real wasserstein_q, oin_real wasserstein_delta, bool return_wasserstein_distance) ->
                std::variant<IndicesValues, std::pair<IndicesValues, oin_real>>
                    {
                      if (return_wasserstein_distance)
                          return opt.match_and_distance(template_dgm, d, wasserstein_q, wasserstein_delta);
                      else
                          return opt.match(template_dgm, d, wasserstein_q, wasserstein_delta);
                    },
                    nb::arg("template_dgm"),
                    nb::arg("dim"),
                    nb::arg("wasserstein_q")=1.0,
                    nb::arg("wasserstein_delta")=0.01,
                    nb::arg("return_wasserstein_distance")=false,
                    nb::call_guard<nb::gil_scoped_release>(),
                    "return target from Wasserstein matching"
            )
            .def_prop_ro("homology_decomposition", &TopologyOptimizer::get_homology_decompostion)
            .def_prop_ro("cohomology_decomposition", &TopologyOptimizer::get_cohomology_decompostion)
            .def("singleton", &TopologyOptimizer::singleton)
            .def("singletons", &TopologyOptimizer::singletons,
                    nb::call_guard<nb::gil_scoped_release>())
            .def("reduce_all", &TopologyOptimizer::reduce_all,
                    nb::call_guard<nb::gil_scoped_release>())
            .def("increase_death", nb::overload_cast<size_t>(&TopologyOptimizer::increase_death, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for increasing death to inf")
            .def("decrease_death", nb::overload_cast<size_t>(&TopologyOptimizer::decrease_death, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for decreasing death to -inf")
            .def("increase_birth", nb::overload_cast<size_t>(&TopologyOptimizer::increase_birth, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for increasing birth to inf")
            .def("decrease_birth", nb::overload_cast<size_t>(&TopologyOptimizer::decrease_birth, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for decreasing birth to -inf")
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, ConflictStrategy)>(&TopologyOptimizer::combine_loss),
                    nb::arg("critical_sets"),
                    nb::arg("strategy"),
                    nb::call_guard<nb::gil_scoped_release>(),
                    "combine critical sets into well-defined assignment of new values to indices")
            .def("update", &TopologyOptimizer::update)
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const TopologyOptimizer& opt) -> TopologyOptimizerStateTuple {
                return std::make_tuple(opt.negate_, opt.fil_, opt.decmp_hom_, opt.decmp_coh_, opt.params_hom_, opt.params_coh_);
            })
            .def("__setstate__", [](TopologyOptimizer& opt, const TopologyOptimizerStateTuple& t) {
                new (&opt) TopologyOptimizer(std::get<1>(t));
                opt.negate_ = std::get<0>(t);
                opt.fil_ = std::get<1>(t);
                opt.decmp_hom_ = std::get<2>(t);
                opt.decmp_coh_ = std::get<3>(t);
                opt.params_hom_ = std::get<4>(t);
                opt.params_coh_ = std::get<5>(t);
            });

}

void init_oineus_top_optimizer(nb::module_& m)
{
    using Simp = oin::Simplex<oin_int>;
    using SimpProd = oin::ProductCell<Simp, Simp>;
    using Cube_1D = oin::Cube<oin_int, 1>;
    using Cube_2D = oin::Cube<oin_int, 2>;
    using Cube_3D = oin::Cube<oin_int, 3>;

    init_oineus_top_optimizer_class<Simp>(m, "TopologyOptimizer", "IndicesValues");
    init_oineus_top_optimizer_class<SimpProd>(m, "TopologyOptimizerProd", "IndicesValuesProd");
    init_oineus_top_optimizer_class<Cube_1D>(m, "TopologyOptimizerCube_1D", "IndicesValuesCube_1D");
    init_oineus_top_optimizer_class<Cube_2D>(m, "TopologyOptimizerCube_2D", "IndicesValuesCube_2D");
    init_oineus_top_optimizer_class<Cube_3D>(m, "TopologyOptimizerCube_3D", "IndicesValuesCube_3D");

    // induced matching
    m.def("get_induced_matching", &oin::get_induced_matching<Simp, oin_real>,
            "Compute induced matching for two filtrations of the same complex",
            nb::arg("included_filtration"),
            nb::arg("containing_filtration"),
            nb::arg("dim")=static_cast<dim_type >(-1),
            nb::arg("n_threads")=1);

    m.def("get_induced_matching", &oin::get_induced_matching<SimpProd, oin_real>,
           "Compute induced matching for two filtrations of the same complex",
           nb::arg("included_filtration"),
           nb::arg("containing_filtration"),
           nb::arg("dim")=static_cast<dim_type >(-1),
           nb::arg("n_threads")=1);

    m.def("get_induced_matching", &oin::get_induced_matching<Cube_1D, oin_real>,
           "Compute induced matching for two filtrations of the same complex",
           nb::arg("included_filtration"),
           nb::arg("containing_filtration"),
           nb::arg("dim")=static_cast<dim_type >(-1),
           nb::arg("n_threads")=1);

    m.def("get_induced_matching", &oin::get_induced_matching<Cube_2D, oin_real>,
           "Compute induced matching for two filtrations of the same complex",
           nb::arg("included_filtration"),
           nb::arg("containing_filtration"),
           nb::arg("dim")=static_cast<dim_type >(-1),
           nb::arg("n_threads")=1);

    m.def("get_induced_matching", &oin::get_induced_matching<Cube_3D, oin_real>,
           "Compute induced matching for two filtrations of the same complex",
           nb::arg("included_filtration"),
           nb::arg("containing_filtration"),
           nb::arg("dim")=static_cast<dim_type >(-1),
           nb::arg("n_threads")=1);
}
