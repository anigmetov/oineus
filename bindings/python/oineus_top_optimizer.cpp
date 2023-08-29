#include "oineus_persistence_bindings.h"

void init_oineus_top_optimizer(py::module& m)
{

    using Real = double;
    using Int = int;

    using Simplex = oin::Simplex<Int, Real>;
    using Filtration = oin::Filtration<Simplex>;
    using TopologyOptimizer = oin::TopologyOptimizer<Simplex>;
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
            .def(py::init<const Filtration&>())
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
            .def("singleton", &TopologyOptimizer::singleton)
            .def("singletons", &TopologyOptimizer::singletons)
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, ConflictStrategy)>(&TopologyOptimizer::combine_loss),
                    py::arg("critical_sets"),
                    py::arg("strategy"),
                    "combine critical sets into well-defined assignment of new values to indices")
            .def("update", &TopologyOptimizer::update);

    // induced matching
    m.def("get_induced_matching", &oin::get_induced_matching<Simplex>,
            "Compute induced matching for two filtrations of the same complex",
            py::arg("included_filtration"),
            py::arg("containing_filtration"),
            py::arg("n_threads")=1);

//    IndicesValues combine_loss(const CriticalSets& critical_sets, ConflictStrategy strategy)
}
