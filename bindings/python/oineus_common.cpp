#include "oineus_persistence_bindings.h"

void init_oineus_common(nb::module_& m)
{
    using VREdge = oin::VREdge<oin_int>;
    using VREdgeStateTuple = std::tuple<decltype(VREdge::x), decltype(VREdge::y)>;

    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;
    using oin::DiagramPlaneDomain;
    using oin::FrechetMeanInit;
    using oin::FiltrationKind;
    using ReductionParams = oin::ReductionParams;
    using KICRParams = oin::KICRParams;
    std::string vr_edge_name = "VREdge";

    // nb::bind_vector<Z2_Column>(m, "Z2_Column");
    // nb::bind_vector<Z2_Matrix>(m, "Z2_Matrix");

    nb::class_<VREdge>(m, vr_edge_name.c_str())
            .def("__init__", [](VREdge* p, oin_int x, oin_int y) {
                new (p) VREdge{x, y};
            }, nb::arg("x")=0, nb::arg("y")=0)
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
            })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const VREdge& p) -> VREdgeStateTuple {
                return std::make_tuple(p.x, p.y);
            })
            .def("__setstate__", [](VREdge& p, const VREdgeStateTuple& t) {
                new (&p) VREdge();
                p.x = std::get<0>(t);
                p.y = std::get<1>(t);
            });

    using RPAdvanced = ReductionParams::Advanced;
    using RPAdvancedTuple = std::tuple<int,    // chunk_size
                                       int,    // col_repr
                                       decltype (RPAdvanced::dims_to_restore_elz)
                                     >;

    using RedParamsTuple = std::tuple<int,    // n_threads
                                      bool,   // use_clearing
                                      bool,   // compute_v
                                      bool,   // compute_u
                                      bool,   // sanity_check
                                      bool,   // verbose
                                      bool,   // use_apparent_pairs
                                      RPAdvanced
                                    >;

    nb::enum_<oin::ColumnRepr>(m, "ColumnRepr", "Working-column data structure used during reduction")
            .value("Set", oin::ColumnRepr::Set, "std::set (baseline, PHAT A-Set)")
            .value("Heap", oin::ColumnRepr::Heap, "lazy max-heap (PHAT A-Heap)")
            .value("Full", oin::ColumnRepr::Full, "dense bitset + max-heap (PHAT A-Full)")
            .value("BitTree", oin::ColumnRepr::BitTree, "hierarchical 64-ary bitset (PHAT A-Bit-Tree, default)");

    using ReductionTimings = oin::ReductionTimings;
    using TimingsStateTuple = std::tuple<double, double, double, double, double, double>;

    nb::class_<ReductionTimings>(m, "ReductionTimings",
            "Per-phase wall-clock breakdown (seconds) of the last reduce() call, "
            "available as Decomposition.timings. Some fields are 0 when a path skips "
            "that phase: the serial path reduces in place, so it has no prepare / "
            "copy_back / copy_pivots. reduction_total is the path-comparable total.")
            .def(nb::init<>())
            .def_rw("prepare", &ReductionTimings::prepare, "build the working atomic-pointer matrix (parallel only)")
            .def_rw("reduce", &ReductionTimings::reduce, "the reduction itself (serial loop or parallel threads)")
            .def_rw("bauer", &ReductionTimings::bauer, "Bauer-trick fill of cleared V columns (only when V materialized under clearing)")
            .def_rw("restore_elz", &ReductionTimings::restore_elz, "ELZ-restore phase (only if dims_to_restore_elz set)")
            .def_rw("copy_back", &ReductionTimings::copy_back, "move working matrix back into r_data/v_data (parallel only)")
            .def_rw("copy_pivots", &ReductionTimings::copy_pivots, "copy pivots into the at-rest pivot array (parallel only)")
            .def_prop_ro("reduction_total", &ReductionTimings::reduction_total,
                    "Total reduction wall-clock across every phase -- comparable across serial and parallel paths.")
            .def_prop_ro("total", &ReductionTimings::total, "Synonym for reduction_total.")
            .def("reset", &ReductionTimings::reset)
            .def("__repr__", [](const ReductionTimings& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def("__getstate__", [](const ReductionTimings& t) -> TimingsStateTuple {
                return std::make_tuple(t.prepare, t.reduce, t.bauer, t.restore_elz, t.copy_back, t.copy_pivots);
            })
            .def("__setstate__", [](ReductionTimings& t, const TimingsStateTuple& s) {
                new (&t) ReductionTimings();
                t.prepare     = std::get<0>(s);
                t.reduce      = std::get<1>(s);
                t.bauer       = std::get<2>(s);
                t.restore_elz = std::get<3>(s);
                t.copy_back   = std::get<4>(s);
                t.copy_pivots = std::get<5>(s);
            });

    using UComputeTimings = oin::UComputeTimings;
    using UTimingsStateTuple = std::tuple<double, double, double, double>;

    nb::class_<UComputeTimings>(m, "UComputeTimings",
            "Per-phase wall-clock breakdown (seconds) of the last compute_u_* call. "
            "Only the fields the chosen strategy uses are nonzero: row-form "
            "(compute_full_u_rows / compute_partial_u_rows) fills transpose_v + "
            "row_solve; column-form (compute_u_from_v / compute_u_from_v_1) fills "
            "col_solve + col_to_row. total is the strategy-comparable U-compute time.")
            .def(nb::init<>())
            .def_rw("transpose_v", &UComputeTimings::transpose_v, "row-form Stage A: build V^T (parallel col->row transpose)")
            .def_rw("row_solve", &UComputeTimings::row_solve, "row-form Stage B: parallel per-row forward substitution")
            .def_rw("col_solve", &UComputeTimings::col_solve, "column-form: solve each U column in parallel")
            .def_rw("col_to_row", &UComputeTimings::col_to_row, "column-form: transpose column-form U into row form")
            .def_prop_ro("total", &UComputeTimings::total, "Total U-compute wall-clock across every phase -- comparable across strategies.")
            .def("reset", &UComputeTimings::reset)
            .def("__repr__", [](const UComputeTimings& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def("__getstate__", [](const UComputeTimings& t) -> UTimingsStateTuple {
                return std::make_tuple(t.transpose_v, t.row_solve, t.col_solve, t.col_to_row);
            })
            .def("__setstate__", [](UComputeTimings& t, const UTimingsStateTuple& s) {
                new (&t) UComputeTimings();
                t.transpose_v = std::get<0>(s);
                t.row_solve   = std::get<1>(s);
                t.col_solve   = std::get<2>(s);
                t.col_to_row  = std::get<3>(s);
            });

    // kwargs ctor defaults are read off a default-constructed instance so the
    // C++ in-class defaults stay the single source of truth (they used to
    // silently differ from the no-arg ctor: n_threads 8 vs 1, chunk_size 256 vs 128)
    const ReductionParams def_rp{};

    nb::class_<RPAdvanced>(m, "ReductionParamsAdvanced",
            "Rarely-tuned reduction knobs, nested as ReductionParams.advanced. "
            "The reduce() kwargs layer still accepts them flat: "
            "dcmp.reduce(chunk_size=256) routes here.")
            .def(nb::init<>())
            .def("__init__",
                [](RPAdvanced* p, int chunk_size, oin::ColumnRepr col_repr, std::vector<dim_type> dims_to_restore_elz) {
                    new (p) RPAdvanced();
                    p->chunk_size = chunk_size;
                    p->col_repr = col_repr;
                    p->dims_to_restore_elz = dims_to_restore_elz;
                }, nb::arg("chunk_size")=def_rp.advanced.chunk_size, nb::arg("col_repr")=def_rp.advanced.col_repr, nb::arg("dims_to_restore_elz")=def_rp.advanced.dims_to_restore_elz)
            .def_rw("chunk_size", &RPAdvanced::chunk_size)
            .def_rw("col_repr", &RPAdvanced::col_repr)
            .def_rw("dims_to_restore_elz", &RPAdvanced::dims_to_restore_elz)
            .def("__repr__", [](const RPAdvanced& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const RPAdvanced& p) -> RPAdvancedTuple {
                    return std::make_tuple(p.chunk_size, static_cast<int>(p.col_repr), p.dims_to_restore_elz);
                })
            .def("__setstate__", [](RPAdvanced& p, const RPAdvancedTuple& t) {
                    new (&p) RPAdvanced();
                    p.chunk_size = std::get<0>(t);
                    p.col_repr = static_cast<oin::ColumnRepr>(std::get<1>(t));
                    p.dims_to_restore_elz = std::get<2>(t);
                });

    nb::class_<ReductionParams>(m, "ReductionParams")
            .def(nb::init<>())
            .def("__init__",
                [](ReductionParams* p, int n_threads, int chunk_size, bool use_clearing, bool compute_v, bool compute_u, std::vector<dim_type> dims_to_restore_elz, oin::ColumnRepr col_repr, bool verbose, bool use_apparent_pairs) {
                    new (p) ReductionParams();
                    p->n_threads = n_threads;
                    p->advanced.chunk_size = chunk_size;
                    p->use_clearing = use_clearing;
                    p->compute_v = compute_v;
                    p->compute_u = compute_u;
                    p->advanced.dims_to_restore_elz = dims_to_restore_elz;
                    p->advanced.col_repr = col_repr;
                    p->verbose = verbose;
                    p->use_apparent_pairs = use_apparent_pairs;
                }, nb::arg("n_threads")=def_rp.n_threads, nb::arg("chunk_size")=def_rp.advanced.chunk_size, nb::arg("use_clearing")=def_rp.use_clearing, nb::arg("compute_v")=def_rp.compute_v, nb::arg("compute_u")=def_rp.compute_u, nb::arg("dims_to_restore_elz")=def_rp.advanced.dims_to_restore_elz, nb::arg("col_repr")=def_rp.advanced.col_repr, nb::arg("verbose")=def_rp.verbose, nb::arg("use_apparent_pairs")=def_rp.use_apparent_pairs)
            .def_rw("n_threads", &ReductionParams::n_threads)
            .def_rw("use_clearing", &ReductionParams::use_clearing)
            .def_rw("compute_v", &ReductionParams::compute_v)
            .def_rw("compute_u", &ReductionParams::compute_u)
            .def_rw("use_apparent_pairs", &ReductionParams::use_apparent_pairs)
            .def_rw("sanity_check", &ReductionParams::sanity_check)
            .def_rw("advanced", &ReductionParams::advanced)
            // back-compat aliases for the renamed fields (read+write, no warning)
            .def_prop_rw("clearing_opt",
                    [](const ReductionParams& p) { return p.use_clearing; },
                    [](ReductionParams& p, bool value) { p.use_clearing = value; },
                    "Deprecated alias for use_clearing.")
            .def_prop_rw("apparent_opt",
                    [](const ReductionParams& p) { return p.use_apparent_pairs; },
                    [](ReductionParams& p, bool value) { p.use_apparent_pairs = value; },
                    "Deprecated alias for use_apparent_pairs.")
            .def_prop_rw("do_sanity_check",
                    [](const ReductionParams& p) { return p.sanity_check; },
                    [](ReductionParams& p, bool value) { p.sanity_check = value; },
                    "Deprecated alias for sanity_check.")
            // timing outputs moved off the params: fail loudly with a pointer to the
            // new location instead of an AttributeError
            .def_prop_ro("elapsed", [](const ReductionParams&) -> double {
                    throw std::runtime_error("ReductionParams.elapsed was removed: reduce() no longer "
                            "writes timings back into the params. Read dcmp.timings.total (or the "
                            "per-phase fields of dcmp.timings) on the reduced Decomposition instead.");
                })
            .def_prop_ro("timings", [](const ReductionParams&) -> ReductionTimings {
                    throw std::runtime_error("ReductionParams.timings was removed: reduce() no longer "
                            "writes timings back into the params. Read dcmp.timings on the reduced "
                            "Decomposition instead.");
                })
            .def_rw("verbose", &ReductionParams::verbose)
            .def("__repr__", [](const ReductionParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const ReductionParams& p) -> RedParamsTuple {
                      return std::make_tuple(p.n_threads,
                              p.use_clearing, p.compute_v, p.compute_u,
                              p.sanity_check, p.verbose,
                              p.use_apparent_pairs, p.advanced);
                    })
            .def("__setstate__", [](ReductionParams& p, const RedParamsTuple& t) {
                    new (&p) ReductionParams();
                      p.n_threads       = std::get<0>(t);
                      p.use_clearing    = std::get<1>(t);
                      p.compute_v       = std::get<2>(t);
                      p.compute_u       = std::get<3>(t);
                      p.sanity_check    = std::get<4>(t);
                      p.verbose         = std::get<5>(t);
                      p.use_apparent_pairs = std::get<6>(t);
                      p.advanced        = std::get<7>(t);
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
            .def_rw("params_im", &KICRParams::params_im)
            .def_rw("params_cok", &KICRParams::params_cok)
            .def("__repr__", [](const KICRParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
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

    nb::enum_<FiltrationKind>(m, "FiltrationKind",
            "Hint set by built-in filtration constructors. Lets oineus.diff "
            "pick reduction defaults and oineus.vis pick rendering "
            "(point cloud vs scalar field) automatically. User is the safe "
            "default for hand-built filtrations.")
            .value("User", FiltrationKind::User)
            .value("Vr", FiltrationKind::Vr)
            .value("Alpha", FiltrationKind::Alpha)
            .value("WeakAlpha", FiltrationKind::WeakAlpha)
            .value("CechDelaunay", FiltrationKind::CechDelaunay)
            .value("Freudenthal", FiltrationKind::Freudenthal)
            .value("Cubical", FiltrationKind::Cubical)
            .value("MinFil", FiltrationKind::MinFil)
            .value("MappingCylinder", FiltrationKind::MappingCylinder)
            .def("as_str", [](const FiltrationKind& self) { return oin::to_string(self); });

    nb::enum_<DiagramPlaneDomain>(m, "DiagramPlaneDomain")
            .value("AboveDiagonal", DiagramPlaneDomain::AboveDiagonal)
            .value("BelowDiagonal", DiagramPlaneDomain::BelowDiagonal)
            .value("Mixed", DiagramPlaneDomain::Mixed)
            .def("as_str", [](const DiagramPlaneDomain& self) {
                switch (self) {
                case DiagramPlaneDomain::AboveDiagonal: return std::string("above");
                case DiagramPlaneDomain::BelowDiagonal: return std::string("below");
                case DiagramPlaneDomain::Mixed: return std::string("mixed");
                default: return std::string("unknown");
                }
            });

    nb::enum_<FrechetMeanInit>(m, "FrechetMeanInit")
            .value("Custom", FrechetMeanInit::Custom)
            .value("FirstDiagram", FrechetMeanInit::FirstDiagram)
            .value("MedoidDiagram", FrechetMeanInit::MedoidDiagram)
            .value("RandomDiagram", FrechetMeanInit::RandomDiagram)
            .value("Grid", FrechetMeanInit::Grid)
            .def("as_str", [](const FrechetMeanInit& self) {
                switch (self) {
                case FrechetMeanInit::Custom: return std::string("custom");
                case FrechetMeanInit::FirstDiagram: return std::string("first");
                case FrechetMeanInit::MedoidDiagram: return std::string("medoid");
                case FrechetMeanInit::RandomDiagram: return std::string("random");
                case FrechetMeanInit::Grid: return std::string("grid");
                default: return std::string("unknown");
                }
            });
}
