#include "oineus_persistence_bindings.h"

void init_oineus_dcmp_stats(nb::module_& m)
{
    using Stats = oin::DecompositionManipStats;

    using StatsStateTuple = std::tuple<double, double, double, double, double, double, double, // elapsed_*
                                       long long, long long, long long, long long, long long,  // counters
                                       size_t, size_t, size_t, size_t>;                        // nnz_*

    nb::class_<Stats>(m, "DecompositionManipStats",
            "Profiling record for decomposition-manipulation methods: per-phase wall-clock "
            "times plus column-operation counts (the headline metric of the vineyards / "
            "move-schedule / warm-start papers) and fill-in (nnz) before/after.")
            .def(nb::init<>())
            .def_rw("elapsed_total", &Stats::elapsed_total)
            .def_rw("elapsed_schedule_build", &Stats::elapsed_schedule_build)
            .def_rw("elapsed_transpose", &Stats::elapsed_transpose)
            .def_rw("elapsed_move", &Stats::elapsed_move)
            .def_rw("elapsed_permute", &Stats::elapsed_permute)
            .def_rw("elapsed_rereduce", &Stats::elapsed_rereduce)
            .def_rw("elapsed_resize", &Stats::elapsed_resize)
            .def_rw("n_transpositions", &Stats::n_transpositions)
            .def_rw("n_moves", &Stats::n_moves)
            .def_rw("n_column_additions_r", &Stats::n_column_additions_r)
            .def_rw("n_column_additions_v", &Stats::n_column_additions_v)
            .def_rw("n_queries", &Stats::n_queries)
            .def_rw("nnz_r_before", &Stats::nnz_r_before)
            .def_rw("nnz_r_after", &Stats::nnz_r_after)
            .def_rw("nnz_v_before", &Stats::nnz_v_before)
            .def_rw("nnz_v_after", &Stats::nnz_v_after)
            .def("n_column_additions", &Stats::n_column_additions,
                    "Total column additions across R and V (n_column_additions_r + n_column_additions_v).")
            .def("reset", &Stats::reset)
            .def("__repr__", [](const Stats& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def("__getstate__", [](const Stats& s) -> StatsStateTuple {
                return std::make_tuple(s.elapsed_total, s.elapsed_schedule_build, s.elapsed_transpose,
                        s.elapsed_move, s.elapsed_permute, s.elapsed_rereduce, s.elapsed_resize,
                        s.n_transpositions, s.n_moves, s.n_column_additions_r, s.n_column_additions_v,
                        s.n_queries, s.nnz_r_before, s.nnz_r_after, s.nnz_v_before, s.nnz_v_after);
            })
            .def("__setstate__", [](Stats& s, const StatsStateTuple& t) {
                new (&s) Stats();
                s.elapsed_total          = std::get<0>(t);
                s.elapsed_schedule_build = std::get<1>(t);
                s.elapsed_transpose      = std::get<2>(t);
                s.elapsed_move           = std::get<3>(t);
                s.elapsed_permute        = std::get<4>(t);
                s.elapsed_rereduce       = std::get<5>(t);
                s.elapsed_resize         = std::get<6>(t);
                s.n_transpositions       = std::get<7>(t);
                s.n_moves                = std::get<8>(t);
                s.n_column_additions_r   = std::get<9>(t);
                s.n_column_additions_v   = std::get<10>(t);
                s.n_queries              = std::get<11>(t);
                s.nnz_r_before           = std::get<12>(t);
                s.nnz_r_after            = std::get<13>(t);
                s.nnz_v_before           = std::get<14>(t);
                s.nnz_v_after            = std::get<15>(t);
            });
}
