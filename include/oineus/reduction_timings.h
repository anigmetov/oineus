#ifndef OINEUS_REDUCTION_TIMINGS_H
#define OINEUS_REDUCTION_TIMINGS_H

#include <iostream>

namespace oineus {

// Per-phase wall-clock breakdown (seconds) of a single VRUDecomposition::reduce()
// call, filled with oineus::Timer. Some fields are legitimately 0 when a code path
// does not run that phase: the serial path reduces in place, so it has no
// prepare / copy_back / copy_pivots; the parallel paths build a working
// (atomic-pointer) matrix (prepare), reduce it (reduce), optionally restore ELZ,
// then move it back into r_data/v_data (copy_back) and copy pivots (copy_pivots).
//
// reduction_total() is the apples-to-apples number to compare across the serial
// and parallel paths. The historical Params::elapsed was NOT comparable: in the
// parallel paths it timed only the reduction core and excluded prepare/copy_back,
// while in the serial path it timed the whole (in-place) reduction. Params::elapsed
// is now kept as a back-compat scalar equal to reduction_total().
struct ReductionTimings {
    double prepare {0.0};       // build the working atomic-pointer matrix; parallel only
    double reduce {0.0};        // the reduction itself (serial loop or parallel threads)
    double restore_elz {0.0};   // ELZ-restore phase; only if dims_to_restore_elz is set
    double copy_back {0.0};     // move working matrix back into r_data/v_data; parallel only
    double copy_pivots {0.0};   // copy pivots into _pivots; parallel only

    // Diagnostic sub-breakdown of `prepare` (the working-matrix build). These are
    // COMPONENTS of prepare and are deliberately NOT summed into reduction_total
    // (that would double-count). boundary_build = raw boundary-matrix build;
    // antitranspose = boundary->coboundary antitranspose (cohomology only; 0 for
    // homology). prepare - boundary_build - antitranspose = the working-column
    // scatter/alloc.
    double boundary_build {0.0};
    double antitranspose {0.0};

    // Total wall-clock of the reduction across every phase -- comparable across paths.
    double reduction_total() const
    {
        return prepare + reduce + restore_elz + copy_back + copy_pivots;
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
    out << ", restore_elz = " << t.restore_elz;
    out << ", copy_back = " << t.copy_back;
    out << ", copy_pivots = " << t.copy_pivots;
    out << ", boundary_build = " << t.boundary_build;
    out << ", antitranspose = " << t.antitranspose << ")";
    return out;
}

} // namespace oineus

#endif // OINEUS_REDUCTION_TIMINGS_H
