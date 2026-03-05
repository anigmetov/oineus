#include "oineus_persistence_bindings.h"

void init_oineus_common(nb::module_& m)
{
    using VREdge = oin::VREdge<oin_int>;

    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;
    using ReductionParams = oin::Params;
    using KICRParams = oin::KICRParams;
    std::string vr_edge_name = "VREdge";

    // nb::bind_vector<Z2_Column>(m, "Z2_Column");
    // nb::bind_vector<Z2_Matrix>(m, "Z2_Matrix");

    nb::class_<VREdge>(m, vr_edge_name.c_str())
            .def(nb::init<oin_int>())
            .def_rw("x", &VREdge::x)
            .def_rw("y", &VREdge::y)
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

    using RedParamsTuple = std::tuple<int,    // n_threads
                                      int,    //chunk_size
                                      bool,   // write_dgms
                                      bool,   // sort_dgms
                                      bool,   // clearing_opt
                                      bool,   // acq_rel
                                      bool,   // print_time
                                      bool,   // compute_v
                                      bool,   // compute_u
                                      bool,   // restore_elz
                                      bool,   // do_sanity_check
                                      double, //elapsed
                                      double, //elapsed_restore_elz
                                      double, //elapsed_copy_back
                                      double, //elapsed_copy_pivots
                                      bool    // verbose
                                    >;

    nb::class_<ReductionParams>(m, "ReductionParams")
            .def(nb::init<>())
            .def("__init__",
                [](ReductionParams* p, int n_threads, int chunk_size, bool clearing_opt, bool compute_v, bool compute_u, bool restore_elz, bool verbose) {
                    new (p) ReductionParams();
                    p->n_threads = n_threads;
                    p->chunk_size = chunk_size;
                    p->clearing_opt = clearing_opt;
                    p->compute_v = compute_v;
                    p->compute_u = compute_u;
                    p->restore_elz = restore_elz;
                    p->verbose = verbose;
                }, nb::arg("n_threads")=8, nb::arg("chunk_size")=256, nb::arg("clearing_opt")=true, nb::arg("compute_v")=false, nb::arg("compute_u")=false, nb::arg("restore_elz")=false, nb::arg("verbose")=false)
            .def_rw("n_threads", &ReductionParams::n_threads)
            .def_rw("chunk_size", &ReductionParams::chunk_size)
            .def_rw("write_dgms", &ReductionParams::write_dgms)
            .def_rw("sort_dgms", &ReductionParams::sort_dgms)
            .def_rw("clearing_opt", &ReductionParams::clearing_opt)
            .def_rw("acq_rel", &ReductionParams::acq_rel)
            .def_rw("print_time", &ReductionParams::print_time)
            .def_rw("elapsed", &ReductionParams::elapsed)
            .def_rw("compute_v", &ReductionParams::compute_v)
            .def_rw("compute_u", &ReductionParams::compute_u)
            .def_rw("restore_elz", &ReductionParams::restore_elz)
            .def_rw("do_sanity_check", &ReductionParams::do_sanity_check)
            .def_rw("elapsed_restore_elz", &ReductionParams::elapsed_restore_elz)
            .def_rw("elapsed_copy_back", &ReductionParams::elapsed_copy_back)
            .def_rw("elapsed_copy_pivots", &ReductionParams::elapsed_copy_pivots)
            .def_rw("verbose", &ReductionParams::verbose)
            .def("__repr__", [](const ReductionParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def("__getstate__", [](const ReductionParams& p) {
                      return std::make_tuple(p.n_threads, p.chunk_size, p.write_dgms,
                              p.sort_dgms, p.clearing_opt, p.acq_rel, p.print_time, p.compute_v, p.compute_u,
                              p.restore_elz, p.do_sanity_check, p.elapsed, p.elapsed_restore_elz,
                              p.elapsed_copy_back, p.elapsed_copy_pivots, p.verbose);
                    })
            .def("__setstate__", [](ReductionParams& p, const RedParamsTuple& t) {
                    new (&p) ReductionParams();
                      p.n_threads       = std::get<0>(t);
                      p.chunk_size      = std::get<1>(t);
                      p.write_dgms      = std::get<2>(t);
                      p.sort_dgms       = std::get<3>(t);
                      p.clearing_opt    = std::get<4>(t);
                      p.acq_rel         = std::get<5>(t);
                      p.print_time      = std::get<6>(t);
                      p.compute_v       = std::get<7>(t);
                      p.compute_u       = std::get<8>(t);
                      p.restore_elz     = std::get<9>(t);
                      p.do_sanity_check = std::get<10>(t);
                      p.elapsed         = std::get<11>(t);
                      p.elapsed_restore_elz = std::get<12>(t);
                      p.elapsed_copy_back = std::get<13>(t);
                      p.elapsed_copy_pivots = std::get<14>(t);
                      p.verbose         = std::get<15>(t);
                    })
    ;

    using KicrStateTuple = std::tuple<bool, bool, bool, bool, bool, bool, bool,
                                     int, ReductionParams, ReductionParams,
                                     ReductionParams, ReductionParams,
                                     ReductionParams>;

    nb::class_<KICRParams>(m, "KICRParams")
            .def(nb::init<>())
            .def("__init__",
                    [](KICRParams* p, bool codomain, bool kernel, bool image, bool cokernel,
                            bool include_zero_persistence, bool verbose, bool sanity_check,
                            int n_threads, const ReductionParams& params_f,
                            const ReductionParams& params_g, const ReductionParams& params_ker,
                            const ReductionParams& params_im, const ReductionParams& params_cok) {
                        new (p) KICRParams();
                        p->codomain = codomain;
                        p->kernel = kernel;
                        p->image = image;
                        p->cokernel = cokernel;
                        p->include_zero_persistence = include_zero_persistence;
                        p->verbose = verbose;
                        p->sanity_check = sanity_check;
                        p->n_threads = n_threads;
                        p->params_f = params_f;
                        p->params_g = params_g;
                        p->params_ker = params_ker;
                        p->params_im = params_im;
                        p->params_cok = params_cok;
                    },
                    nb::arg("codomain")=false,
                    nb::arg("kernel")=true,
                    nb::arg("image")=true,
                    nb::arg("cokernel")=true,
                    nb::arg("include_zero_persistence")=false,
                    nb::arg("verbose")=false,
                    nb::arg("sanity_check")=false,
                    nb::arg("n_threads")=1,
                    nb::arg("params_f")=ReductionParams(),
                    nb::arg("params_g")=ReductionParams(),
                    nb::arg("params_ker")=ReductionParams(),
                    nb::arg("params_im")=ReductionParams(),
                    nb::arg("params_cok")=ReductionParams())
            .def_rw("codomain", &KICRParams::codomain)
            .def_rw("kernel", &KICRParams::kernel)
            .def_rw("image", &KICRParams::image)
            .def_rw("cokernel", &KICRParams::cokernel)
            .def_rw("include_zero_persistence", &KICRParams::include_zero_persistence)
            .def_rw("verbose", &KICRParams::verbose)
            .def_rw("sanity_check", &KICRParams::sanity_check)
            .def_rw("n_threads", &KICRParams::n_threads)
            .def_rw("params_f", &KICRParams::params_f)
            .def_rw("params_g", &KICRParams::params_g)
            .def_rw("params_ker", &KICRParams::params_ker)
            .def_rw("params_im", &KICRParams::params_ker)
            .def_rw("params_cok", &KICRParams::params_ker)
            .def("__repr__", [](const KICRParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def("__getstate__", [](const KICRParams& p) {
                      return std::make_tuple(p.codomain, p.kernel, p.image, p.cokernel, p.include_zero_persistence, p.verbose, p.sanity_check,
                              p.n_threads, p.params_f, p.params_g, p.params_ker, p.params_im, p.params_cok);
                    })
            .def("__setstate__",
                    [](KICRParams& p, const KicrStateTuple& t) {
                          new (&p) KICRParams();
                          p.codomain = std::get<0>(t);
                          p.kernel = std::get<1>(t);
                          p.image = std::get<2>(t);
                          p.cokernel = std::get<3>(t);
                          p.include_zero_persistence = std::get<4>(t);
                          p.verbose = std::get<5>(t);
                          p.sanity_check = std::get<6>(t);
                          p.n_threads = std::get<7>(t);
                          p.params_f = std::get<8>(t);
                          p.params_g = std::get<9>(t);
                          p.params_ker = std::get<10>(t);
                          p.params_im = std::get<11>(t);
                          p.params_cok = std::get<12>(t);
                    })
    ;

    nb::enum_<DenoiseStrategy>(m, "DenoiseStrategy")
            .value("BirthBirth", DenoiseStrategy::BirthBirth, "(b, d) maps to (b, b)")
            .value("DeathDeath", DenoiseStrategy::DeathDeath, "(b, d) maps to (d, d)")
            .value("Midway", DenoiseStrategy::Midway, "((b, d) maps to ((b+d)/2, (b+d)/2)")
            .def("as_str", [](const DenoiseStrategy& self) { return denoise_strategy_to_string(self); });

    nb::enum_<ConflictStrategy>(m, "ConflictStrategy")
            .value("Max", ConflictStrategy::Max, "choose maximal displacement")
            .value("Avg", ConflictStrategy::Avg, "average gradients")
            .value("Sum", ConflictStrategy::Sum, "sum gradients")
            .value("FixCritAvg", ConflictStrategy::FixCritAvg, "use matching on critical, average gradients on other cells")
            .def("as_str", [](const ConflictStrategy& self) { return conflict_strategy_to_string(self); });
}
