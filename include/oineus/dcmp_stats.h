#ifndef OINEUS_DCMP_STATS_H
#define OINEUS_DCMP_STATS_H

#include <cstddef>
#include <iostream>

namespace oineus {

// Profiling/benchmarking record for the decomposition-manipulation methods
// (vineyards, moves, move schedules, Luo-Nelson warm-start updates).
//
// Two tiers of information:
//  * wall-clock per phase (seconds), filled with oineus::Timer;
//  * operation counters -- the headline metric used by both papers is the
//    number of column operations, not wall-clock, so we count column
//    additions on R and V separately, plus transpositions/moves and cheap
//    queries. nnz_* record fill-in (matrix non-zeros) before/after.
//
// A pointer to one of these is threaded (optionally) through every
// manipulation method; nullptr means "do not collect".
struct DecompositionManipStats {
    // wall-clock per phase, in seconds
    double elapsed_total {0.0};
    double elapsed_schedule_build {0.0};   // LCS/LIS + permutation prep
    double elapsed_transpose {0.0};        // vineyard transposition execution
    double elapsed_move {0.0};             // move (MoveLeft/MoveRight) execution
    double elapsed_permute {0.0};          // Pr/Pc row-relabel application (Alg 2/3)
    double elapsed_rereduce {0.0};         // re-reduction passes (Alg 2/3)
    double elapsed_resize {0.0};           // Alg 3 insert/delete reshaping

    // operation counters (the papers' headline metric)
    long long n_transpositions {0};
    long long n_moves {0};
    long long n_column_additions_r {0};    // add_to_column applied to r_data
    long long n_column_additions_v {0};    // add_to_column applied to v_data
    long long n_queries {0};               // pivot / membership lookups

    // fill-in metrics (sum of column sizes)
    std::size_t nnz_r_before {0}, nnz_r_after {0};
    std::size_t nnz_v_before {0}, nnz_v_after {0};

    void reset() { *this = DecompositionManipStats{}; }

    // convenience: total column operations across R and V
    long long n_column_additions() const
    {
        return n_column_additions_r + n_column_additions_v;
    }
};

inline std::ostream& operator<<(std::ostream& out, const DecompositionManipStats& s)
{
    out << "DecompositionManipStats(";
    out << "n_col_ops = " << s.n_column_additions();
    out << " (r = " << s.n_column_additions_r << ", v = " << s.n_column_additions_v << ")";
    out << ", n_transpositions = " << s.n_transpositions;
    out << ", n_moves = " << s.n_moves;
    out << ", n_queries = " << s.n_queries;
    out << ", elapsed_total = " << s.elapsed_total << "s";
    out << " (schedule = " << s.elapsed_schedule_build;
    out << ", transpose = " << s.elapsed_transpose;
    out << ", move = " << s.elapsed_move;
    out << ", permute = " << s.elapsed_permute;
    out << ", rereduce = " << s.elapsed_rereduce;
    out << ", resize = " << s.elapsed_resize << ")";
    out << ", nnz_r = " << s.nnz_r_before << " -> " << s.nnz_r_after;
    out << ", nnz_v = " << s.nnz_v_before << " -> " << s.nnz_v_after;
    out << ")";
    return out;
}

} // namespace oineus

#endif // OINEUS_DCMP_STATS_H
