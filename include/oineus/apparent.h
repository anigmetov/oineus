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

namespace oineus {

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
template<class Fil>
ApparentMatching<typename Fil::Int>
detect_apparent_local(const Fil& fil)
{
    using Int = typename Fil::Int;
    const std::size_t n = fil.size();

    ApparentMatching<Int> am;
    am.init(n);

    const auto& cells = fil.cells();

    for(std::size_t c = 0; c < n; ++c) {
        const auto& cell = cells[c];
        if (cell.dim() == 0)
            continue;

        // youngest facet f* = max sorted_id over facets of cell
        Int fstar = Int(-1);
        for(const auto& fuid : cell.get_cell().boundary()) {
            Int sid = fil.get_sorted_id_by_uid(fuid);
            if (sid > fstar)
                fstar = sid;
        }

        // oldest cofacet of f* = min sorted_id over cofacets of f*
        Int oldest = std::numeric_limits<Int>::max();
        for(const auto& cuid : cells[fstar].get_cell().coboundary()) {
            Int sid = fil.get_sorted_id_by_uid(cuid);
            if (sid < oldest)
                oldest = sid;
        }

        if (oldest == static_cast<Int>(c))
            am.add(fstar, static_cast<Int>(c));
    }

    return am;
}

} // namespace oineus
