#pragma once

// Apparent pairs (Bauer, Ripser). An apparent pair (sigma, tau) in a cellwise
// filtration with total order sorted_id and boundary matrix D (columns = cells,
// rows = cells, D[r][c]=1 iff cell r is a facet of cell c) satisfies
//   - sigma = low(D_tau): sigma is the youngest facet of tau (max-sorted_id
//     entry of column tau), AND
//   - tau is the oldest cofacet of sigma: sigma appears in no earlier column.
// Such a pair is a persistence pair, already reduced before reduction starts,
// and field-independent. For VR/cubical/Freudenthal filtrations apparent pairs
// are typically ~90-99% of all cells; not materializing their columns is the
// prize (see agents_outputs/APPARENT_PAIRS_PLAN.md and the plan file).
//
// This header (Stage 0) provides the matching type and the detectors used as
// oracles. The fused build-time skip + reduction resolver are wired in later
// stages.

#include <vector>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <ostream>

#include "sparse_matrix.h"

namespace oineus {

// Apparent detection is column-centric and needs BOTH facets and cofacets of a
// cell, i.e. a cell type exposing coboundary() (Cube does; Simplex does not).
// The fused factory uses this to `if constexpr`-gate the apparent build path so it
// is never instantiated for unsupported cell types (e.g. VR / Simplex).
template<class Cell, class = void>
struct SupportsApparent : std::false_type {};

// The slim Cube's coboundary takes the shared geometry (the GridDomain owned by
// the filtration), so probe coboundary(geometry) rather than a no-argument
// coboundary(). Cube matches; Simplex (no coboundary at all) does not.
template<class Cell>
struct SupportsApparent<Cell, std::void_t<decltype(std::declval<const Cell&>().coboundary(std::declval<const typename Cell::Geometry&>()))>>
        : std::true_type {};

// Detects the fused RV working-column type (RVColumn<Int,2>, which carries both an
// r_column and a v_column) so the reducer's apparent-resolver hook is compiled only
// for the RV path -- the R-only reduction never builds a decorated matrix, and its
// column type has no nested ::Column.
template<class T, class = void>
struct IsRVColumn : std::false_type {};

template<class T>
struct IsRVColumn<T, std::void_t<decltype(std::declval<T&>().r_column),
                                 decltype(std::declval<T&>().v_column)>> : std::true_type {};

// partner record for an index-space [0,N). Index space is the matrix's own
// order: sorted_id for the homology boundary, the antitransposed/reversed order
// for the cohomology coboundary. The detectors below are agnostic to which.
template<class Int_>
struct ApparentMatching {
    using Int = Int_;

    // apparent_pivot_of_row[r] = c if (r,c) is apparent (r = pivot/birth row,
    // c = death column), else -1. This is exactly a _pivots pre-seed.
    std::vector<Int> apparent_pivot_of_row;
    // is_apparent_col[c] = 1 if column c is an apparent (death) column whose
    // working slot is left null.
    std::vector<char> is_apparent_col;
    std::size_t n_apparent {0};

    void init(std::size_t n)
    {
        apparent_pivot_of_row.assign(n, Int(-1));
        is_apparent_col.assign(n, 0);
        n_apparent = 0;
    }

    bool is_apparent(Int c) const
    {
        return c >= 0 && static_cast<std::size_t>(c) < is_apparent_col.size() && is_apparent_col[c];
    }

    bool empty() const { return n_apparent == 0; }

    void add(Int pivot_row, Int col)
    {
        apparent_pivot_of_row[pivot_row] = col;
        is_apparent_col[col] = 1;
        ++n_apparent;
    }
};

template<class Int>
std::ostream& operator<<(std::ostream& os, const ApparentMatching<Int>& am)
{
    os << "ApparentMatching(n_apparent=" << am.n_apparent
       << ", size=" << am.is_apparent_col.size() << ")";
    return os;
}

// Regenerates the working R column of an apparent (null) matrix column on demand,
// in the matrix's own index space, via a caller-supplied closure (which knows the
// filtration, the homology/cohomology direction, and the cell's direct
// (co)boundary). Const + no shared mutable state => thread-safe; the caller owns
// the returned column. A null `matching`/`resolve` (the default) means "no
// apparent optimization", so the reduction's resolver hook is a no-op branch.
template<class Int>
struct ApparentResolver {
    const ApparentMatching<Int>*               matching {nullptr};
    const std::function<SparseColumn<Int>(Int)>* resolve {nullptr};

    bool active() const { return matching != nullptr && resolve != nullptr && not matching->empty(); }
    bool is_apparent(Int c) const { return matching != nullptr && matching->is_apparent(c); }
    SparseColumn<Int> resolve_r(Int c) const { return (*resolve)(c); }
};

template<class Int>
std::ostream& operator<<(std::ostream& os, const ApparentResolver<Int>& r)
{
    os << "ApparentResolver(active=" << (r.active() ? "true" : "false");
    if (r.matching != nullptr)
        os << ", n_apparent=" << r.matching->n_apparent;
    os << ")";
    return os;
}

// Generic O(nnz) detector on a column-major matrix (each column sorted
// ascending). One left-to-right pass with a per-row "first column seen" flag:
// (low, c) is apparent iff c is the first column whose support contains low.
// Works on any boundary/coboundary matrix; result is in that matrix's index
// space.
template<class Matrix>
ApparentMatching<typename Matrix::value_type::value_type>
detect_apparent_generic(const Matrix& M)
{
    using Int = typename Matrix::value_type::value_type;
    const std::size_t n = M.size();

    ApparentMatching<Int> am;
    am.init(n);

    std::vector<Int> seen_first_col(n, Int(-1));   // first column containing row r

    for(std::size_t c = 0; c < n; ++c) {
        const auto& col = M[c];
        if (col.empty())
            continue;
        Int low = col.back();
        for(Int r : col)
            if (seen_first_col[r] < 0)
                seen_first_col[r] = static_cast<Int>(c);
        // low first appeared in column c  <=>  c is the oldest cofacet of low
        if (seen_first_col[low] == static_cast<Int>(c))
            am.add(low, static_cast<Int>(c));
    }

    return am;
}

// O(n * nnz) brute-force oracle: for each column c with low = low(M_c), check no
// earlier column contains row low. Slow; for tests only.
template<class Matrix>
ApparentMatching<typename Matrix::value_type::value_type>
detect_apparent_bruteforce(const Matrix& M)
{
    using Int = typename Matrix::value_type::value_type;
    const std::size_t n = M.size();

    ApparentMatching<Int> am;
    am.init(n);

    for(std::size_t c = 0; c < n; ++c) {
        const auto& col = M[c];
        if (col.empty())
            continue;
        Int low = col.back();
        bool oldest = true;
        for(std::size_t cp = 0; cp < c && oldest; ++cp)
            if (std::binary_search(M[cp].begin(), M[cp].end(), low))
                oldest = false;
        if (oldest)
            am.add(low, static_cast<Int>(c));
    }

    return am;
}

// Local per-cell detector for cells exposing BOTH boundary() and coboundary()
// (e.g. Cube). Column-centric: for each cell tau, take its youngest facet
// f* = argmax sorted_id over facets, then tau is apparent iff tau is the oldest
// cofacet of f* (argmin sorted_id over cofacets of f*). Operates in sorted_id
// (homology) space; the cohomology index relabeling is layered on by the fused
// builder. Requires a complete complex (all cofacets present) -- caller must
// ensure not is_subfiltration().
//
// Parallel over cells. The writes are conflict-free without any synchronization:
// is_apparent_col[c] is written only by task c, and apparent_pivot_of_row[fstar]
// is written only by the unique apparent cell whose youngest facet is fstar (a
// facet has at most one apparent cofacet -- its oldest one -- so two distinct
// apparent tasks never target the same fstar slot). n_apparent is tallied after
// the parallel pass. This matters for performance: detection touches every cell
// (boundary + coboundary), so a serial pass would dominate (and erase) the
// build-time win of skipping the apparent columns.
template<class Fil>
ApparentMatching<typename Fil::Int>
detect_apparent_local(const Fil& fil, int n_threads = 1)
{
    using Int = typename Fil::Int;
    const std::size_t n = fil.size();

    ApparentMatching<Int> am;
    am.init(n);

    const auto& cells = fil.cells();

    tf::Executor executor(std::max(1, n_threads));
    tf::Taskflow taskflow;
    taskflow.for_each_index((std::size_t)0, n, (std::size_t)1,
            [&am, &cells, &fil](std::size_t c) {
                const auto& cell = cells[c];
                if (cell.dim() == 0)
                    return;

                // youngest facet f* = max sorted_id over facets of cell
                Int fstar = Int(-1);
                for(const auto& fuid : cell.get_cell().boundary(fil.geometry())) {
                    Int sid = fil.get_sorted_id_by_uid(fuid);
                    if (sid > fstar)
                        fstar = sid;
                }

                // oldest cofacet of f* = min sorted_id over cofacets of f*
                Int oldest = std::numeric_limits<Int>::max();
                for(const auto& cuid : cells[fstar].get_cell().coboundary(fil.geometry())) {
                    Int sid = fil.get_sorted_id_by_uid(cuid);
                    if (sid < oldest)
                        oldest = sid;
                }

                if (oldest == static_cast<Int>(c)) {
                    am.is_apparent_col[c] = 1;
                    am.apparent_pivot_of_row[fstar] = static_cast<Int>(c);
                }
            });
    executor.run(taskflow).get();

    std::size_t cnt = 0;
    for(char v : am.is_apparent_col)
        cnt += static_cast<std::size_t>(v);
    am.n_apparent = cnt;

    return am;
}

// Apparent matching in the REDUCTION's matrix-index space, dualize-aware. The
// detector itself (detect_apparent_local) is field- and direction-independent and
// works in sorted_id (homology) space, identifying each apparent pair as
// (b = youngest facet / birth, d = death cell). This wrapper maps those pairs into
// whichever matrix the reduction will actually reduce:
//   - homology   (dualize=false): the matrix index IS the sorted_id, so the null
//     column is the death d and the pivot row is the birth b -- returned as-is.
//   - cohomology (dualize=true): the boundary is antitransposed, so cell sorted_id
//     s sits at matrix index N-1-s, and birth/death swap: the null column is the
//     BIRTH cell (matrix index N-1-b) and the pivot row is the DEATH cell (N-1-d).
// This matches emit_column_'s convention exactly, so a _pivots pre-seed built from
// the result drives diagram extraction with no further change.
template<class Fil>
ApparentMatching<typename Fil::Int>
detect_apparent_matrix(const Fil& fil, bool dualize, int n_threads = 1)
{
    using Int = typename Fil::Int;

    ApparentMatching<Int> hom = detect_apparent_local(fil, n_threads);
    if (not dualize)
        return hom;

    const std::size_t n = fil.size();
    ApparentMatching<Int> am;
    am.init(n);
    for(std::size_t b = 0; b < n; ++b) {
        Int d = hom.apparent_pivot_of_row[b];
        if (d < 0)
            continue;
        Int null_col  = static_cast<Int>(fil.index_in_matrix(b, true));                       // N-1-b
        Int pivot_row = static_cast<Int>(fil.index_in_matrix(static_cast<std::size_t>(d), true)); // N-1-d
        am.add(pivot_row, null_col);
    }
    return am;
}

} // namespace oineus
