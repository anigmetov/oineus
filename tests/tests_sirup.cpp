#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <oineus/oineus.h>

// Tests for VRUDecomposition::remove_simplices (SiRUP, Giunti-Lazovskis):
// update R = D V after removing a coface-closed set of cells, checked against a
// from-scratch reduction of the restricted boundary matrix.

using Int = int;
using Decomp = oineus::VRUDecomposition<Int>;
using MatrixData = Decomp::MatrixData;
using Col = MatrixData::value_type;

// index-level pairing { (low(R[c]), c) : R[c] != 0 } -- the invariant SiRUP must
// preserve relative to a from-scratch reduction (R itself is not unique).
static std::set<std::pair<Int, Int>> pairing(const MatrixData& R)
{
    std::set<std::pair<Int, Int>> s;
    for(size_t c = 0; c < R.size(); ++c)
        if (not R[c].empty())
            s.emplace(R[c].back(), static_cast<Int>(c));
    return s;
}

// coface up-closure of `seeds` under the facet relation encoded in D.
static std::vector<size_t> coface_closure(const MatrixData& D, const std::vector<size_t>& seeds)
{
    const size_t n = D.size();
    std::vector<std::vector<size_t>> cofacets(n);
    for(size_t c = 0; c < n; ++c)
        for(Int f : D[c])
            cofacets[static_cast<size_t>(f)].push_back(c);
    std::vector<char> in(n, 0);
    std::vector<size_t> stack;
    for(size_t s : seeds)
        if (not in[s]) { in[s] = 1; stack.push_back(s); }
    while(not stack.empty()) {
        size_t c = stack.back(); stack.pop_back();
        for(size_t u : cofacets[c])
            if (not in[u]) { in[u] = 1; stack.push_back(u); }
    }
    std::vector<size_t> out;
    for(size_t c = 0; c < n; ++c)
        if (in[c]) out.push_back(c);
    return out;
}

// boundary of the survivors, with rows/columns renumbered to the compacted order
static MatrixData compact_boundary(const MatrixData& D, const std::vector<size_t>& removed)
{
    const size_t n = D.size();
    std::set<size_t> dead(removed.begin(), removed.end());
    std::vector<size_t> remap(n, n);
    size_t p = 0;
    for(size_t c = 0; c < n; ++c)
        if (not dead.count(c))
            remap[c] = p++;
    MatrixData out(p);
    for(size_t c = 0; c < n; ++c) {
        if (dead.count(c)) continue;
        Col col;
        for(Int r : D[c])
            col.push_back(static_cast<Int>(remap[static_cast<size_t>(r)]));
        std::sort(col.begin(), col.end());
        out[remap[c]] = std::move(col);
    }
    return out;
}

static MatrixData reduce_pairing_only(const MatrixData& D)
{
    Decomp dc(D);
    oineus::Params p;
    p.compute_v = true;
    p.clearing_opt = false;
    p.n_threads = 1;
    dc.reduce(p);
    return dc.r_data;
}

static Decomp reduced(const MatrixData& D, bool clearing)
{
    Decomp dc(D);
    oineus::Params p;
    p.compute_v = true;
    p.clearing_opt = clearing;
    p.n_threads = 1;
    dc.reduce(p);
    return dc;
}

static bool is_matrix_reduced(const MatrixData& R)
{
    std::set<Int> lows;
    for(const auto& col : R)
        if (not col.empty())
            if (not lows.insert(col.back()).second)
                return false;
    return true;
}

// Triangle: vertices 0,1,2; edges 3={0,1},4={0,2},5={1,2}; triangle 6={3,4,5}.
static MatrixData triangle_boundary()
{
    MatrixData d(7);
    d[3] = {0, 1};
    d[4] = {0, 2};
    d[5] = {1, 2};
    d[6] = {3, 4, 5};
    return d;
}

using Fil = oineus::Filtration<oineus::Simplex<Int>, double>;

static Fil grid_filtration(size_t side, unsigned seed)
{
    using Grid = oineus::Grid<Int, double, 2>;
    using GridPoint = typename Grid::GridPoint;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> data(side * side);
    for(auto& x : data) x = dist(gen);
    GridPoint dims{static_cast<Int>(side), static_cast<Int>(side)};
    Grid grid(dims, false, data.data(), Grid::DataLocation::VERTEX);
    return grid.freudenthal_filtration(2, false, 1);
}

static Decomp reduced_fil(const Fil& fil, bool clearing)
{
    Decomp dc(fil, false);                 // per-dimension blocks, like real usage
    oineus::Params p;
    p.compute_v = true;
    p.clearing_opt = clearing;
    p.n_threads = 1;
    dc.reduce(p);
    return dc;
}

// the core check: SiRUP pairing == from-scratch pairing of the survivor
// filtration; reduced; and R = D V for the clearing-off case. Built from a
// filtration so the decomposition carries real per-dimension blocks.
static void check_removal_fil(Fil fil, const std::vector<size_t>& seeds, bool clearing)
{
    auto D = fil.boundary_matrix(1);
    auto L = coface_closure(D, seeds);
    auto expected = pairing(reduce_pairing_only(compact_boundary(D, L)));

    Decomp dc = reduced_fil(fil, clearing);
    oineus::DecompositionManipStats stats;
    dc.remove_simplices(L, &stats, 1);

    REQUIRE(dc.r_data.size() == D.size() - L.size());
    REQUIRE(is_matrix_reduced(dc.r_data));
    REQUIRE(pairing(dc.r_data) == expected);
    if (not clearing)
        REQUIRE(dc.is_reduced_consistent());     // R = D V, V need not be upper-tri
}

// boundary-matrix path (single dimension block) -- kept for the triangle case.
static void check_removal(const MatrixData& D, const std::vector<size_t>& seeds, bool clearing)
{
    auto L = coface_closure(D, seeds);
    auto expected = pairing(reduce_pairing_only(compact_boundary(D, L)));

    Decomp dc = reduced(D, clearing);
    oineus::DecompositionManipStats stats;
    dc.remove_simplices(L, &stats, 1);

    REQUIRE(dc.r_data.size() == D.size() - L.size());
    REQUIRE(is_matrix_reduced(dc.r_data));
    REQUIRE(pairing(dc.r_data) == expected);
    if (not clearing)
        REQUIRE(dc.is_reduced_consistent());
}

TEST_CASE("remove_simplices: triangle, every single-cell star")
{
    auto D = triangle_boundary();
    for(size_t c = 0; c < D.size(); ++c)
        for(bool clearing : {false, true})
            check_removal(D, {c}, clearing);
}

TEST_CASE("remove_simplices: grid filtration, random coface-closed removals")
{
    for(unsigned seed = 1; seed <= 12; ++seed) {
        auto fil = grid_filtration(5, seed);
        std::mt19937 gen(100 + seed);
        std::uniform_int_distribution<size_t> pick(0, fil.size() - 1);
        for(int trial = 0; trial < 6; ++trial) {
            std::vector<size_t> seeds{pick(gen), pick(gen), pick(gen)};
            for(bool clearing : {false, true})
                check_removal_fil(fil, seeds, clearing);
        }
    }
}

TEST_CASE("remove_simplices: repeated successive removals stay correct")
{
    // Keep a parallel boundary matrix D_cur compacted in lockstep with the
    // decomposition; each round picks a seed in the *current* matrix order, so
    // the two stay aligned and the from-scratch oracle is always D_cur reduced.
    Fil fil = grid_filtration(5, 42);
    MatrixData D_cur = fil.boundary_matrix(1);
    Decomp dc = reduced_fil(fil, false);
    std::mt19937 gen(7);

    for(int round = 0; round < 4 and D_cur.size() > 2; ++round) {
        std::uniform_int_distribution<size_t> pick(0, D_cur.size() - 1);
        std::vector<size_t> seeds{pick(gen)};
        auto L = coface_closure(D_cur, seeds);

        dc.remove_simplices(L, nullptr, 1);
        D_cur = compact_boundary(D_cur, L);

        REQUIRE(dc.r_data.size() == D_cur.size());
        REQUIRE(is_matrix_reduced(dc.r_data));
        REQUIRE(pairing(dc.r_data) == pairing(reduce_pairing_only(D_cur)));
        REQUIRE(dc.is_reduced_consistent());
    }
}

TEST_CASE("remove_simplices: rejects a non-coface-closed set")
{
    auto D = triangle_boundary();
    Decomp dc = reduced(D, false);
    // {edge 3} alone is not coface-closed: the triangle 6 has it as a face.
    REQUIRE_THROWS(dc.remove_simplices(std::vector<size_t>{3}, nullptr, 1));
}
