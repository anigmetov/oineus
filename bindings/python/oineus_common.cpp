#include "oineus_persistence_bindings.h"

void init_oineus_common(py::module& m)
{
    using namespace pybind11::literals;

    using VREdge = oin::VREdge<oin_int>;

    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;
    using ReductionParams = oin::Params;
    using KICRParams = oin::KICRParams;
    std::string vr_edge_name = "VREdge";

    py::class_<VREdge>(m, vr_edge_name.c_str())
            .def(py::init<oin_int>())
            .def_readwrite("x", &VREdge::x)
            .def_readwrite("y", &VREdge::y)
            .def("__getitem__", [](const VREdge& p, int i) {
              if (i == 0)
                  return p.x;
              else if (i == 1)
                  return p.y;
              else
                  throw std::out_of_range("i must be 0 or 1");
            })
            .def("__repr__", [](const VREdge& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<ReductionParams>(m, "ReductionParams")
            .def(py::init<>())
            .def_readwrite("n_threads", &ReductionParams::n_threads)
            .def_readwrite("chunk_size", &ReductionParams::chunk_size)
            .def_readwrite("write_dgms", &ReductionParams::write_dgms)
            .def_readwrite("sort_dgms", &ReductionParams::sort_dgms)
            .def_readwrite("clearing_opt", &ReductionParams::clearing_opt)
            .def_readwrite("acq_rel", &ReductionParams::acq_rel)
            .def_readwrite("print_time", &ReductionParams::print_time)
            .def_readwrite("elapsed", &ReductionParams::elapsed)
            .def_readwrite("compute_v", &ReductionParams::compute_v)
            .def_readwrite("compute_u", &ReductionParams::compute_u)
            .def_readwrite("do_sanity_check", &ReductionParams::do_sanity_check)
            .def_readwrite("verbose", &ReductionParams::verbose)
            .def("__repr__", [](const ReductionParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(py::pickle(
                    // __getstate__

                    [](const ReductionParams& p) {
                      return py::make_tuple(p.n_threads, p.chunk_size, p.write_dgms,
                              p.sort_dgms, p.clearing_opt, p.acq_rel, p.print_time, p.compute_v, p.compute_u,
                              p.do_sanity_check, p.elapsed, p.verbose);
                    },
                    // __setstate__
                    [](py::tuple t) {
                      if (t.size() != 12)
                          throw std::runtime_error("Invalid tuple for ReductionParams");

                      ReductionParams p;

                      int i = 0;

                      p.n_threads = t[i++].cast<decltype(p.n_threads)>();
                      p.chunk_size = t[i++].cast<decltype(p.chunk_size)>();
                      p.write_dgms = t[i++].cast<decltype(p.write_dgms)>();
                      p.sort_dgms = t[i++].cast<decltype(p.sort_dgms)>();
                      p.clearing_opt = t[i++].cast<decltype(p.clearing_opt)>();
                      p.acq_rel = t[i++].cast<decltype(p.acq_rel)>();
                      p.print_time = t[i++].cast<decltype(p.print_time)>();
                      p.compute_v = t[i++].cast<decltype(p.compute_v)>();
                      p.compute_u = t[i++].cast<decltype(p.compute_u)>();
                      p.do_sanity_check = t[i++].cast<decltype(p.do_sanity_check)>();

                      p.elapsed = t[i++].cast<decltype(p.elapsed)>();
                      p.verbose = t[i++].cast<decltype(p.verbose)>();

                      return p;
                    }));

    py::class_<KICRParams>(m, "KICRParams")
            .def(py::init<>())
            .def_readwrite("codomain", &KICRParams::codomain)
            .def_readwrite("kernel", &KICRParams::kernel)
            .def_readwrite("image", &KICRParams::image)
            .def_readwrite("cokernel", &KICRParams::cokernel)
            .def_readwrite("include_zero_persistence", &KICRParams::include_zero_persistence)
            .def_readwrite("verbose", &KICRParams::verbose)
            .def_readwrite("sanity_check", &KICRParams::sanity_check)
            .def_readwrite("n_threads", &KICRParams::n_threads)
            .def_readwrite("params_f", &KICRParams::params_f)
            .def_readwrite("params_g", &KICRParams::params_g)
            .def_readwrite("params_ker", &KICRParams::params_ker)
            .def_readwrite("params_im", &KICRParams::params_ker)
            .def_readwrite("params_cok", &KICRParams::params_ker)
            .def("__repr__", [](const KICRParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(py::pickle(
                    // __getstate__

                    [](const KICRParams& p) {
                      return py::make_tuple(p.codomain, p.kernel, p.image, p.cokernel, p.include_zero_persistence, p.verbose, p.sanity_check,
                              p.n_threads, p.params_f, p.params_g, p.params_ker, p.params_im, p.params_cok);
                    },
                    // __setstate__
                    [](py::tuple t) {
                      if (t.size() != 13)
                          throw std::runtime_error("Invalid tuple for KICRParams");

                      KICRParams p;

                      int i = 0;

                      p.codomain = t[i++].cast<decltype(p.codomain)>();
                      p.kernel = t[i++].cast<decltype(p.kernel)>();
                      p.image = t[i++].cast<decltype(p.image)>();
                      p.cokernel = t[i++].cast<decltype(p.cokernel)>();
                      p.include_zero_persistence = t[i++].cast<decltype(p.include_zero_persistence)>();
                      p.verbose = t[i++].cast<decltype(p.verbose)>();
                      p.sanity_check = t[i++].cast<decltype(p.sanity_check)>();
                      p.n_threads = t[i++].cast<decltype(p.n_threads)>();
                      p.params_f = t[i++].cast<decltype(p.params_f)>();
                      p.params_g = t[i++].cast<decltype(p.params_g)>();
                      p.params_ker = t[i++].cast<decltype(p.params_ker)>();
                      p.params_im = t[i++].cast<decltype(p.params_im)>();
                      p.params_cok = t[i++].cast<decltype(p.params_cok)>();

                      return p;
                    }));

    py::enum_<DenoiseStrategy>(m, "DenoiseStrategy", py::arithmetic())
            .value("BirthBirth", DenoiseStrategy::BirthBirth, "(b, d) maps to (b, b)")
            .value("DeathDeath", DenoiseStrategy::DeathDeath, "(b, d) maps to (d, d)")
            .value("Midway", DenoiseStrategy::Midway, "((b, d) maps to ((b+d)/2, (b+d)/2)")
            .def("as_str", [](const DenoiseStrategy& self) { return denoise_strategy_to_string(self); });

    py::enum_<ConflictStrategy>(m, "ConflictStrategy", py::arithmetic())
            .value("Max", ConflictStrategy::Max, "choose maximal displacement")
            .value("Avg", ConflictStrategy::Avg, "average gradients")
            .value("Sum", ConflictStrategy::Sum, "sum gradients")
            .value("FixCritAvg", ConflictStrategy::FixCritAvg, "use matching on critical, average gradients on other cells")
            .def("as_str", [](const ConflictStrategy& self) { return conflict_strategy_to_string(self); });


}