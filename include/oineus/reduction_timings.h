#ifndef OINEUS_REDUCTION_TIMINGS_H
#define OINEUS_REDUCTION_TIMINGS_H

#include <iostream>

namespace oineus {

// Per-phase wall-clock breakdown (seconds) of a single VRUDecomposition::reduce()
// call, filled with oineus::Timer and stored on the decomposition (timings_).
// Some fields are legitimately 0 when a code path does not run that phase: the
// serial path reduces in place, so it has no prepare / copy_back / copy_pivots;
// the parallel paths build a working (atomic-pointer) matrix (prepare), reduce it
// (reduce), optionally restore ELZ, then move it back into r_data/v_data
// (copy_back) and copy pivots (copy_pivots).
//
// reduction_total() is the apples-to-apples number to compare across the serial
// and parallel paths.
struct ReductionTimings {
    double prepare {0.0};       // build the working atomic-pointer matrix; parallel only
    double reduce {0.0};        // the reduction itself (serial loop or parallel threads)
    double bauer {0.0};         // Bauer-trick fill of cleared V columns (V[s] = R[piv(s)]);
                                // only when V is materialized under clearing (do_restore / keep_working)
    double restore_elz {0.0};   // ELZ-restore phase; only if dims_to_restore_elz is set
    double copy_back {0.0};     // move working matrix back into r_data/v_data; parallel only
    double copy_pivots {0.0};   // copy pivots into _pivots; parallel only

    // Total wall-clock of the reduction across every phase -- comparable across paths.
    double reduction_total() const
    {
        return prepare + reduce + bauer + restore_elz + copy_back + copy_pivots;
    }

    // Synonym for reduction_total().
    double total() const { return reduction_total(); }

    void reset() { *this = ReductionTimings{}; }
};

inline std::ostream& operator<<(std::ostream& out, const ReductionTimings& t)
{
    out << "ReductionTimings(total = " << t.reduction_total() << "s";
    out << ", prepare = " << t.prepare;
    out << ", reduce = " << t.reduce;
    out << ", bauer = " << t.bauer;
    out << ", restore_elz = " << t.restore_elz;
    out << ", copy_back = " << t.copy_back;
    out << ", copy_pivots = " << t.copy_pivots << ")";
    return out;
}

// Per-phase wall-clock breakdown (seconds) of a single U-computation call on a
// reduced VRUDecomposition. Each compute_u_* method resets this and fills only
// the fields its strategy uses, so the unused fields stay 0:
//   - row-form (V^T U^T = Id; compute_full_u_rows / compute_partial_u_rows):
//     transpose_v (build V^T once, parallel) + row_solve (parallel forward subst).
//   - column-form (R u_c = D_c via compute_u_from_v, or V u_c = e_c via
//     compute_u_from_v_1): col_solve (solve each U column, parallel) +
//     col_to_row (transpose the column-form U into the at-rest row form u_data_t).
// total() is the apples-to-apples U-compute wall time regardless of strategy.
struct UComputeTimings {
    double transpose_v {0.0};  // row-form Stage A: build V^T (parallel col->row transpose)
    double row_solve   {0.0};  // row-form Stage B: parallel per-row forward substitution
    double col_solve   {0.0};  // column-form: solve each U column in parallel
    double col_to_row  {0.0};  // column-form: transpose column-form U into row form

    double total() const { return transpose_v + row_solve + col_solve + col_to_row; }

    void reset() { *this = UComputeTimings{}; }
};

inline std::ostream& operator<<(std::ostream& out, const UComputeTimings& t)
{
    out << "UComputeTimings(total = " << t.total() << "s";
    out << ", transpose_v = " << t.transpose_v;
    out << ", row_solve = " << t.row_solve;
    out << ", col_solve = " << t.col_solve;
    out << ", col_to_row = " << t.col_to_row << ")";
    return out;
}

} // namespace oineus

#endif // OINEUS_REDUCTION_TIMINGS_H
