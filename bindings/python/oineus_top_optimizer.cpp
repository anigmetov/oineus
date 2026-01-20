#include "oineus_persistence_bindings.h"

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
            .def("match", [](TopologyOptimizer& opt, Diagram& template_dgm, dim_type d, oin_real wasserstein_q, bool return_wasserstein_distance) -> std::variant<IndicesValues, std::pair<IndicesValues, oin_real>>
                    {
                      if (return_wasserstein_distance)
                          return opt.match_and_distance(template_dgm, d, wasserstein_q);
                      else
                          return opt.match(template_dgm, d, wasserstein_q);
                    },
                    nb::arg("template_dgm"),
                    nb::arg("dim"),
                    nb::arg("wasserstein_q")=1.0,
                    nb::arg("return_wasserstein_distance")=false,
                    "return target from Wasserstein matching"
            )
            .def_prop_ro("homology_decomposition", &TopologyOptimizer::get_homology_decompostion)
            .def_prop_ro("cohomology_decomposition", &TopologyOptimizer::get_cohomology_decompostion)
            .def("singleton", &TopologyOptimizer::singleton)
            .def("singletons", &TopologyOptimizer::singletons)
            .def("reduce_all", &TopologyOptimizer::reduce_all)
            .def("increase_death", nb::overload_cast<size_t>(&TopologyOptimizer::increase_death, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for increasing death to inf")
            .def("decrease_death", nb::overload_cast<size_t>(&TopologyOptimizer::decrease_death, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for decreasing death to -inf")
            .def("increase_birth", nb::overload_cast<size_t>(&TopologyOptimizer::increase_birth, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for increasing birth to inf")
            .def("decrease_birth", nb::overload_cast<size_t>(&TopologyOptimizer::decrease_birth, nb::const_), nb::arg("negative_simplex_idx"), "return critical set for decreasing birth to -inf")
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, ConflictStrategy)>(&TopologyOptimizer::combine_loss),
                    nb::arg("critical_sets"),
                    nb::arg("strategy"),
                    "combine critical sets into well-defined assignment of new values to indices")
            .def("update", &TopologyOptimizer::update);

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
}
