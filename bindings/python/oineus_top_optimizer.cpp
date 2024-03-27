#include "oineus_persistence_bindings.h"

void init_oineus_top_optimizer(py::module& m)
{

    using Real = double;
    using Int = int;

    using Simp = oin::Simplex<Int>;
    using SWV = oin::CellWithValue<oin::Simplex<Int>, Real>;
    using SimplexFiltration = oin::Filtration<Simp, Real>;
    using TopologyOptimizer = oin::TopologyOptimizer<Simp, Real>;
    using Indices = typename TopologyOptimizer::Indices;
    using Values = typename TopologyOptimizer::Values;
    using IndicesValues = typename TopologyOptimizer::IndicesValues;
    using CrititcalSet = typename TopologyOptimizer::CriticalSet;
    using Target = typename TopologyOptimizer::Target;
    using Indices = typename TopologyOptimizer::Indices;
    using Values = typename TopologyOptimizer::Values;
    using CriticalSets = typename TopologyOptimizer::CriticalSets;
    using ConflictStrategy = oin::ConflictStrategy;
    using Diagram = typename oin::Diagrams<Real>::Dgm;

    py::class_<IndicesValues>(m, "IndicesValues")
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
    py::class_<TopologyOptimizer>(m, "TopologyOptimizer")
            .def(py::init<const SimplexFiltration&>())
            .def("compute_diagram", [](TopologyOptimizer& opt, bool include_inf_points) { return PyOineusDiagrams<Real>(opt.compute_diagram(include_inf_points)); },
                    py::arg("include_inf_points"),
                    "compute diagrams in all dimensions")
            .def("simplify", &TopologyOptimizer::simplify,
                    py::arg("epsilon"),
                    py::arg("strategy"),
                    py::arg("dim"), "make points with persistence less than epsilon go to the diagonal")
            .def("get_nth_persistence", &TopologyOptimizer::get_nth_persistence,
                    py::arg("dim"), py::arg("n"), "return n-th persistence value in d-dimensional persistence diagram")
            .def("match", [](TopologyOptimizer& opt, Diagram& template_dgm, dim_type d, Real wasserstein_q, bool return_wasserstein_distance) -> std::variant<IndicesValues, std::pair<IndicesValues, Real>>
                    {
                      if (return_wasserstein_distance)
                          return opt.match_and_distance(template_dgm, d, wasserstein_q);
                      else
                          return opt.match(template_dgm, d, wasserstein_q);
                    },
                    py::arg("template_dgm"),
                    py::arg("dim"),
                    py::arg("wasserstein_q")=1.0,
                    py::arg("return_wasserstein_distance")=false,
                    "return target from Wasserstein matching"
            )
            .def_property_readonly("homology_decomposition", &TopologyOptimizer::get_homology_decompostion)
            .def_property_readonly("cohomology_decomposition", &TopologyOptimizer::get_cohomology_decompostion)
            .def("singleton", &TopologyOptimizer::singleton)
            .def("singletons", &TopologyOptimizer::singletons)
            .def("reduce_all", &TopologyOptimizer::reduce_all)
            .def("increase_death", py::overload_cast<size_t>(&TopologyOptimizer::increase_death, py::const_), py::arg("negative_simplex_idx"), "return critical set for increasing death to inf")
            .def("decrease_death", py::overload_cast<size_t>(&TopologyOptimizer::decrease_death, py::const_), py::arg("negative_simplex_idx"), "return critical set for decreasing death to -inf")
            .def("increase_birth", py::overload_cast<size_t>(&TopologyOptimizer::increase_birth, py::const_), py::arg("negative_simplex_idx"), "return critical set for increasing birth to inf")
            .def("decrease_birth", py::overload_cast<size_t>(&TopologyOptimizer::decrease_birth, py::const_), py::arg("negative_simplex_idx"), "return critical set for decreasing birth to -inf")
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, ConflictStrategy)>(&TopologyOptimizer::combine_loss),
                    py::arg("critical_sets"),
                    py::arg("strategy"),
                    "combine critical sets into well-defined assignment of new values to indices")
            .def("linear_decrease_death", &TopologyOptimizer::linear_decrease_death, "linear decrease death")
            .def("linear_increase_death", &TopologyOptimizer::linear_increase_death, "linear increase death")
            .def("linear_decrease_birth", &TopologyOptimizer::linear_decrease_death, "linear decrease birth")
            .def("linear_increase_birth", &TopologyOptimizer::linear_increase_death, "linear increase birth")
            .def("linear_increase_deaths", &TopologyOptimizer::linear_increase_deaths, "linear increase death")
            .def("linear_decrease_deaths", &TopologyOptimizer::linear_decrease_deaths, "linear increase death")
            .def("linear_increase_births", &TopologyOptimizer::linear_increase_births, "linear increase death")
            .def("linear_decrease_births", &TopologyOptimizer::linear_decrease_births, "linear increase death")
            .def("update", &TopologyOptimizer::update);

    // induced matching
    m.def("get_induced_matching", &oin::get_induced_matching<Simp, Real>,
            "Compute induced matching for two filtrations of the same complex",
            py::arg("included_filtration"),
            py::arg("containing_filtration"),
            py::arg("dim")=static_cast<dim_type >(-1),
            py::arg("n_threads")=1);
}
