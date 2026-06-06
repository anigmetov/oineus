#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <oineus/oineus.h>

using dim_type = size_t;
using Int = int;
using Decomp = oineus::VRUDecomposition<Int>;
using MatrixData = Decomp::MatrixData;

// ---------------------------------------------------------------------------
// Oracles / helpers
// ---------------------------------------------------------------------------

// Index-level persistence pairing: { (low(R[c]), c) : R[c] != 0 }. This is the
// invariant a manipulation must preserve relative to a from-scratch reduction
// of the same (permuted) boundary matrix; the actual R/V need not be identical
// because reduction is not unique, but the pairing is.
static std::set<std::pair<Int, Int>> pairing(const Decomp& d)
{
    std::set<std::pair<Int, Int>> s;
    for(size_t c = 0; c < d.r_data.size(); ++c)
        if (not d.r_data[c].empty())
            s.emplace(d.r_data[c].back(), static_cast<Int>(c));
    return s;
}

static std::vector<size_t> invert(const std::vector<size_t>& p)
{
    std::vector<size_t> inv(p.size());
    for(size_t i = 0; i < p.size(); ++i)
        inv[p[i]] = i;
    return inv;
}

// O(n^2) longest strictly-increasing subsequence length (test oracle).
static size_t lis_len(const std::vector<size_t>& a)
{
    if (a.empty())
        return 0;
    std::vector<size_t> dp(a.size(), 1);
    size_t best = 1;
    for(size_t i = 0; i < a.size(); ++i) {
        for(size_t j = 0; j < i; ++j)
            if (a[j] < a[i])
                dp[i] = std::max(dp[i], dp[j] + 1);
        best = std::max(best, dp[i]);
    }
    return best;
}

// Optimal move-schedule size: n - sum of per-dimension LIS.
static size_t expected_min_moves(const std::vector<size_t>& new_to_old,
                                 const std::vector<Int>& dim_first,
                                 const std::vector<Int>& dim_last)
{
    auto old_to_new = invert(new_to_old);
    size_t total_lis = 0;
    for(size_t d = 0; d < dim_first.size(); ++d) {
        std::vector<size_t> seq;
        for(size_t p = static_cast<size_t>(dim_first[d]); p <= static_cast<size_t>(dim_last[d]); ++p)
            seq.push_back(old_to_new[p]);
        total_lis += lis_len(seq);
    }
    return new_to_old.size() - total_lis;
}

// new column k = (old column new_to_old[k]) with row entries relabeled by
// old_to_new, kept sorted.
static MatrixData permute_boundary(const MatrixData& d, const std::vector<size_t>& new_to_old)
{
    auto old_to_new = invert(new_to_old);
    MatrixData out(d.size());
    for(size_t k = 0; k < d.size(); ++k) {
        const auto& src = d[new_to_old[k]];
        std::vector<Int> col;
        col.reserve(src.size());
        for(Int r : src)
            col.push_back(static_cast<Int>(old_to_new[static_cast<size_t>(r)]));
        std::sort(col.begin(), col.end());
        out[k] = std::move(col);
    }
    return out;
}

static Decomp reduce_from_scratch(const MatrixData& d)
{
    Decomp dc(d);
    oineus::Params p;
    p.compute_v = true;
    p.clearing_opt = false;
    p.n_threads = 1;
    dc.reduce(p);
    return dc;
}

static MatrixData identity_matrix(size_t n)
{
    MatrixData m(n);
    for(size_t i = 0; i < n; ++i)
        m[i] = {static_cast<Int>(i)};
    return m;
}

// Triangle: vertices 0,1,2 (dim 0); edges 3={0,1}, 4={0,2}, 5={1,2} (dim 1);
// triangle 6={3,4,5} (dim 2).
static MatrixData triangle_boundary()
{
    MatrixData d(7);
    d[0] = {}; d[1] = {}; d[2] = {};
    d[3] = {0, 1};
    d[4] = {0, 2};
    d[5] = {1, 2};
    d[6] = {3, 4, 5};
    return d;
}

// A small lower-star Freudenthal filtration on a random 2D grid, reduced with V.
static Decomp grid_decomp(size_t side, unsigned seed)
{
    using Grid = oineus::Grid<Int, double, 2>;
    using GridPoint = typename Grid::GridPoint;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> data(side * side);
    for(auto& x : data)
        x = dist(gen);

    GridPoint dims{static_cast<Int>(side), static_cast<Int>(side)};
    Grid grid(dims, false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.freudenthal_filtration(2, false, 1);

    Decomp dc(fil, false);
    oineus::Params p;
    p.compute_v = true;
    p.clearing_opt = false;
    p.n_threads = 1;
    dc.reduce(p);
    return dc;
}

// ---------------------------------------------------------------------------
// reduce_column_with_pivots_ : reproduces the standard reduction
// ---------------------------------------------------------------------------
TEST_CASE("reduce_column_with_pivots_ reproduces standard reduction")
{
    for(auto D : {triangle_boundary(), grid_decomp(5, 1).d_data}) {
        Decomp standard = reduce_from_scratch(D);

        Decomp manual(D);
        manual.r_data = D;
        manual.v_data = identity_matrix(D.size());
        manual._pivots.assign(D.size(), -1);
        long long ar = 0, av = 0;
        for(size_t c = 0; c < D.size(); ++c)
            manual.reduce_column_with_pivots_(manual.r_data, manual.v_data, c, manual._pivots, ar, av);
        manual.is_reduced = true;

        REQUIRE(pairing(manual) == pairing(standard));
        REQUIRE(manual.sanity_check());
    }
}

// ---------------------------------------------------------------------------
// transpose : single adjacent transposition
// ---------------------------------------------------------------------------
TEST_CASE("transpose single adjacent pair (triangle)")
{
    auto D = triangle_boundary();

    // valid same-dimension adjacent pairs: vertices (0,1),(1,2); edges (3,4),(4,5)
    for(size_t i : {size_t(0), size_t(1), size_t(3), size_t(4)}) {
        Decomp dc = reduce_from_scratch(D);
        std::vector<size_t> new_to_old(D.size());
        std::iota(new_to_old.begin(), new_to_old.end(), 0);
        std::swap(new_to_old[i], new_to_old[i + 1]);

        dc.transpose(i);

        REQUIRE(dc.sanity_check());
        REQUIRE(dc.d_data == permute_boundary(D, new_to_old));
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
    }
}

// ---------------------------------------------------------------------------
// transpose : exhaustive valid adjacent pairs on a small grid
// ---------------------------------------------------------------------------
TEST_CASE("transpose exhaustive adjacent pairs on grid")
{
    Decomp base = grid_decomp(4, 7);
    const MatrixData D0 = base.d_data;
    const size_t n = D0.size();

    size_t n_valid = 0;
    for(size_t i = 0; i + 1 < n; ++i) {
        // valid iff sigma_i not a face of sigma_{i+1}
        if (std::binary_search(D0[i + 1].begin(), D0[i + 1].end(), static_cast<Int>(i)))
            continue;
        ++n_valid;

        Decomp dc = reduce_from_scratch(D0);
        dc.transpose(i);

        std::vector<size_t> new_to_old(n);
        std::iota(new_to_old.begin(), new_to_old.end(), 0);
        std::swap(new_to_old[i], new_to_old[i + 1]);

        REQUIRE(dc.sanity_check());
        REQUIRE(dc.d_data == permute_boundary(D0, new_to_old));
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
    }
    REQUIRE(n_valid > 0);
}

// ---------------------------------------------------------------------------
// transpose_to : random within-dimension permutations on a grid
// ---------------------------------------------------------------------------
TEST_CASE("transpose_to random within-dimension permutation")
{
    for(unsigned seed = 0; seed < 5; ++seed) {
        Decomp base = grid_decomp(5, 100 + seed);
        const MatrixData D0 = base.d_data;
        const size_t n = D0.size();

        // build a permutation that only shuffles within each dimension block
        std::vector<size_t> new_to_old(n);
        std::iota(new_to_old.begin(), new_to_old.end(), 0);
        std::mt19937 gen(seed + 1);
        for(size_t d = 0; d < base.dim_first.size(); ++d) {
            size_t lo = static_cast<size_t>(base.dim_first[d]);
            size_t hi = static_cast<size_t>(base.dim_last[d]);
            std::shuffle(new_to_old.begin() + lo, new_to_old.begin() + hi + 1, gen);
        }

        Decomp dc = reduce_from_scratch(D0);
        oineus::DecompositionManipStats stats;
        size_t n_transp = dc.transpose_to(new_to_old, &stats);

        REQUIRE(dc.sanity_check());
        REQUIRE(dc.d_data == permute_boundary(D0, new_to_old));
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
        REQUIRE(static_cast<long long>(n_transp) == stats.n_transpositions);
    }
}

// Rebuild row-incidence indices from the matrices and compare to the ones the
// decomposition maintains in dynamic mode.
static bool ri_consistent(const Decomp& d)
{
    auto build = [&](const MatrixData& m) {
        MatrixData ri(d.n_rows);
        for(size_t c = 0; c < m.size(); ++c)
            for(Int r : m[c])
                ri[static_cast<size_t>(r)].push_back(static_cast<Int>(c));
        return ri;   // already sorted: columns visited in increasing order
    };
    return d.ri_r_ == build(d.r_data) and d.ri_v_ == build(d.v_data) and d.ri_d_ == build(d.d_data);
}

// ---------------------------------------------------------------------------
// dynamic mode: the row index stays consistent through chained transpositions
// ---------------------------------------------------------------------------
TEST_CASE("dynamic row index stays consistent under transpositions")
{
    Decomp dc = grid_decomp(5, 314);
    dc.make_dynamic(2);
    REQUIRE(dc.is_dynamic());
    REQUIRE(ri_consistent(dc));

    std::mt19937 gen(7);
    const MatrixData D0 = dc.d_data;
    for(int step = 0; step < 40; ++step) {
        size_t n = dc.r_data.size();
        size_t i = gen() % (n - 1);
        // only transpose a valid (same-dimension) adjacent pair
        if (std::binary_search(dc.d_data[i + 1].begin(), dc.d_data[i + 1].end(), static_cast<Int>(i)))
            continue;
        dc.transpose(i);
        REQUIRE(ri_consistent(dc));
    }
    REQUIRE(dc.sanity_check());
    REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
}

// new_to_old for Move(i,j): cell at i goes to j; cells in between shift by one.
static std::vector<size_t> move_perm(size_t n, size_t i, size_t j)
{
    std::vector<size_t> new_to_old(n);
    std::iota(new_to_old.begin(), new_to_old.end(), 0);
    if (i < j) {
        for(size_t p = i; p < j; ++p) new_to_old[p] = p + 1;
        new_to_old[j] = i;
    } else if (i > j) {
        for(size_t p = i; p > j; --p) new_to_old[p] = p - 1;
        new_to_old[j] = i;
    }
    return new_to_old;
}

// ---------------------------------------------------------------------------
// move : single move within a dimension block
// ---------------------------------------------------------------------------
TEST_CASE("move within dimension block")
{
    Decomp base = grid_decomp(5, 42);
    const MatrixData D0 = base.d_data;
    const size_t n = D0.size();

    // pick the dim-1 block and move its first cell to the last and vice versa
    size_t d = 1;
    size_t lo = static_cast<size_t>(base.dim_first[d]);
    size_t hi = static_cast<size_t>(base.dim_last[d]);
    REQUIRE(hi > lo);

    for(auto pr : {std::make_pair(lo, hi), std::make_pair(hi, lo)}) {
        Decomp dc = reduce_from_scratch(D0);
        dc.move(pr.first, pr.second);
        auto p = move_perm(n, pr.first, pr.second);
        REQUIRE(dc.sanity_check());
        REQUIRE(dc.d_data == permute_boundary(D0, p));
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
    }
}

// ---------------------------------------------------------------------------
// move : stress test, many random within-dimension moves both directions
// (the donor / conjugate-once implementation, validated per move)
// ---------------------------------------------------------------------------
TEST_CASE("move stress random both directions")
{
    for(unsigned seed = 0; seed < 4; ++seed) {
        Decomp dc = grid_decomp(5, 500 + seed);
        std::mt19937 gen(seed + 123);

        for(int step = 0; step < 12; ++step) {
            // pick a dimension block with >= 2 cells and two distinct positions in it
            size_t d = gen() % dc.dim_first.size();
            size_t lo = static_cast<size_t>(dc.dim_first[d]);
            size_t hi = static_cast<size_t>(dc.dim_last[d]);
            if (hi <= lo)
                continue;
            size_t a = lo + gen() % (hi - lo + 1);
            size_t b = lo + gen() % (hi - lo + 1);
            if (a == b)
                continue;
            dc.move(a, b);
            REQUIRE(dc.sanity_check());
            REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
        }
    }
}

// ---------------------------------------------------------------------------
// apply_move_schedule : random within-dimension permutations on a grid
// ---------------------------------------------------------------------------
TEST_CASE("apply_move_schedule random within-dimension permutation")
{
    for(unsigned seed = 0; seed < 5; ++seed) {
        Decomp base = grid_decomp(5, 200 + seed);
        const MatrixData D0 = base.d_data;
        const size_t n = D0.size();

        std::vector<size_t> new_to_old(n);
        std::iota(new_to_old.begin(), new_to_old.end(), 0);
        std::mt19937 gen(seed + 17);
        size_t n_nonidentity = 0;
        for(size_t d = 0; d < base.dim_first.size(); ++d) {
            size_t lo = static_cast<size_t>(base.dim_first[d]);
            size_t hi = static_cast<size_t>(base.dim_last[d]);
            std::shuffle(new_to_old.begin() + lo, new_to_old.begin() + hi + 1, gen);
        }
        for(size_t p = 0; p < n; ++p)
            if (new_to_old[p] != p) ++n_nonidentity;

        Decomp dc = reduce_from_scratch(D0);
        oineus::DecompositionManipStats stats;
        size_t n_moves = dc.apply_move_schedule(new_to_old, &stats);

        REQUIRE(dc.sanity_check());
        REQUIRE(dc.d_data == permute_boundary(D0, new_to_old));
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
        REQUIRE(static_cast<long long>(n_moves) == stats.n_moves);
        // schedule must be of optimal size n - sum|LIS| (Piekenbrock Alg 4/5)
        REQUIRE(n_moves == expected_min_moves(new_to_old, base.dim_first, base.dim_last));
        REQUIRE(n_moves <= n_nonidentity);
    }
}

// ---------------------------------------------------------------------------
// apply_move_schedule : minimality on a known rotation (single dim block)
// ---------------------------------------------------------------------------
TEST_CASE("apply_move_schedule minimal on a rotation")
{
    // four dim-0 cells (empty boundary) -> one dimension block of size 4
    MatrixData D(4);  // all empty columns
    Decomp dc = reduce_from_scratch(D);

    // new_to_old = [1,2,3,0]: cell 0 rotates to the back -- optimal is 1 move
    std::vector<size_t> new_to_old{1, 2, 3, 0};
    oineus::DecompositionManipStats st;
    size_t n_moves = dc.apply_move_schedule(new_to_old, &st);

    REQUIRE(n_moves == 1);
    REQUIRE(dc.d_data == permute_boundary(D, new_to_old));
}

// ---------------------------------------------------------------------------
// Luo-Nelson Alg 2 : warm-start update under a pure reorder
// ---------------------------------------------------------------------------
TEST_CASE("update_with_permutation (Luo-Nelson Alg 2) random within-dimension permutation")
{
    for(unsigned seed = 0; seed < 5; ++seed) {
        Decomp base = grid_decomp(5, 300 + seed);
        const MatrixData D0 = base.d_data;
        const size_t n = D0.size();

        std::vector<size_t> new_to_old(n);
        std::iota(new_to_old.begin(), new_to_old.end(), 0);
        std::mt19937 gen(seed + 31);
        for(size_t d = 0; d < base.dim_first.size(); ++d) {
            size_t lo = static_cast<size_t>(base.dim_first[d]);
            size_t hi = static_cast<size_t>(base.dim_last[d]);
            std::shuffle(new_to_old.begin() + lo, new_to_old.begin() + hi + 1, gen);
        }

        Decomp dc = reduce_from_scratch(D0);
        oineus::DecompositionManipStats stats;
        dc.update_with_permutation(new_to_old, &stats);

        REQUIRE(dc.is_reduced_consistent());
        REQUIRE(dc.sanity_check());           // V is re-triangularized into the new order
        REQUIRE(dc.d_data == permute_boundary(D0, new_to_old));
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(dc.d_data)));
    }
}

// Identity permutation should leave a reduced decomposition essentially
// untouched (warm start does no column additions).
TEST_CASE("update_with_permutation identity is a no-op reduction")
{
    Decomp base = grid_decomp(5, 999);
    const MatrixData D0 = base.d_data;
    const size_t n = D0.size();
    std::vector<size_t> identity(n);
    std::iota(identity.begin(), identity.end(), 0);

    Decomp dc = reduce_from_scratch(D0);
    oineus::DecompositionManipStats stats;
    dc.update_with_permutation(identity, &stats);

    REQUIRE(dc.is_reduced_consistent());
    REQUIRE(dc.d_data == D0);
    REQUIRE(stats.n_column_additions_r == 0);
    REQUIRE(pairing(dc) == pairing(base));
}

// Build the new-order boundary for a survivors-only new_to_old (all entries
// are old indices; faces must all survive -- coface-closed deletion).
static MatrixData build_survivor_boundary(const MatrixData& D0, const std::vector<long long>& new_to_old)
{
    const size_t n_new = new_to_old.size();
    std::vector<long long> old_to_new(D0.size(), -1);
    for(size_t k = 0; k < n_new; ++k)
        old_to_new[static_cast<size_t>(new_to_old[k])] = static_cast<long long>(k);
    MatrixData out(n_new);
    for(size_t k = 0; k < n_new; ++k) {
        std::vector<Int> col;
        for(Int r : D0[static_cast<size_t>(new_to_old[k])]) {
            long long nr = old_to_new[static_cast<size_t>(r)];
            REQUIRE(nr >= 0);   // face must survive
            col.push_back(static_cast<Int>(nr));
        }
        std::sort(col.begin(), col.end());
        out[k] = std::move(col);
    }
    return out;
}

static int cell_dim(size_t o, const std::vector<Int>& df, const std::vector<Int>& dl)
{
    for(size_t d = 0; d < df.size(); ++d)
        if (static_cast<Int>(o) >= df[d] and static_cast<Int>(o) <= dl[d])
            return static_cast<int>(d);
    return -1;
}

// derive dim_first/dim_last from a dimension-sorted per-cell dim vector
static void derive_dims(const std::vector<int>& dims, std::vector<Int>& ndf, std::vector<Int>& ndl)
{
    int maxd = 0;
    for(int d : dims)
        maxd = std::max(maxd, d);
    ndf.assign(maxd + 1, 0);
    ndl.assign(maxd + 1, -1);
    std::vector<char> seen(maxd + 1, 0);
    for(size_t k = 0; k < dims.size(); ++k) {
        int d = dims[k];
        if (not seen[d]) { ndf[d] = static_cast<Int>(k); seen[d] = 1; }
        ndl[d] = static_cast<Int>(k);
    }
}

// ---------------------------------------------------------------------------
// Luo-Nelson Alg 3 : deletion of top cells + reorder of survivors
// ---------------------------------------------------------------------------
TEST_CASE("update_with_edits delete top cells and reorder")
{
    for(unsigned seed = 0; seed < 4; ++seed) {
        Decomp base = grid_decomp(4, 400 + seed);
        const MatrixData D0 = base.d_data;
        const size_t n_old = D0.size();
        const size_t top_dim = base.dim_first.size() - 1;
        const size_t top_lo = static_cast<size_t>(base.dim_first[top_dim]);
        const size_t top_hi = static_cast<size_t>(base.dim_last[top_dim]);

        // delete the last few top-dimensional cells (always coface-closed)
        std::mt19937 gen(seed + 71);
        size_t n_top = top_hi - top_lo + 1;
        size_t n_del = 1 + (gen() % std::max<size_t>(1, n_top / 2));
        std::vector<char> deleted(n_old, 0);
        for(size_t t = 0; t < n_del; ++t)
            deleted[top_hi - t] = 1;

        // survivors in old order, then shuffle within each dimension block
        std::vector<long long> new_to_old;
        for(size_t o = 0; o < n_old; ++o)
            if (not deleted[o])
                new_to_old.push_back(static_cast<long long>(o));
        // shuffle within dimension blocks of the survivors
        size_t pos = 0;
        for(size_t d = 0; d < base.dim_first.size(); ++d) {
            size_t lo = static_cast<size_t>(base.dim_first[d]);
            size_t hi = static_cast<size_t>(base.dim_last[d]);
            size_t cnt = 0;
            for(size_t o = lo; o <= hi; ++o) if (not deleted[o]) ++cnt;
            if (cnt > 1)
                std::shuffle(new_to_old.begin() + pos, new_to_old.begin() + pos + cnt, gen);
            pos += cnt;
        }

        MatrixData newB = build_survivor_boundary(D0, new_to_old);
        std::vector<int> dims;
        for(long long o : new_to_old)
            dims.push_back(cell_dim(static_cast<size_t>(o), base.dim_first, base.dim_last));
        std::vector<Int> ndf, ndl;
        derive_dims(dims, ndf, ndl);

        Decomp dc = reduce_from_scratch(D0);
        oineus::DecompositionManipStats st;
        dc.update_with_edits(new_to_old, newB, ndf, ndl, &st);

        REQUIRE(dc.r_data.size() == new_to_old.size());
        REQUIRE(dc.sanity_check());
        REQUIRE(dc.d_data == newB);
        REQUIRE(dc.dim_first == ndf);
        REQUIRE(dc.dim_last == ndl);
        REQUIRE(pairing(dc) == pairing(reduce_from_scratch(newB)));
    }
}

// ---------------------------------------------------------------------------
// Luo-Nelson Alg 3 : insertion of new cells
// ---------------------------------------------------------------------------
TEST_CASE("update_with_edits insert cells at end")
{
    Decomp base = grid_decomp(4, 55);
    const MatrixData D0 = base.d_data;
    const size_t n_old = D0.size();
    const size_t top_dim = base.dim_first.size() - 1;
    const Int e0 = base.dim_first[1];   // first edge index

    // append two new top-dimensional cells (boundary = three existing edges),
    // which keeps the order dimension-blocked
    std::vector<long long> new_to_old(n_old);
    std::iota(new_to_old.begin(), new_to_old.end(), 0);
    new_to_old.push_back(-1);
    new_to_old.push_back(-1);

    MatrixData newB = D0;
    newB.push_back(std::vector<Int>{e0, e0 + 1, e0 + 2});
    newB.push_back(std::vector<Int>{e0 + 1, e0 + 2, e0 + 3});

    std::vector<int> dims;
    for(size_t o = 0; o < n_old; ++o)
        dims.push_back(cell_dim(o, base.dim_first, base.dim_last));
    dims.push_back(static_cast<int>(top_dim));
    dims.push_back(static_cast<int>(top_dim));
    std::vector<Int> ndf, ndl;
    derive_dims(dims, ndf, ndl);

    Decomp dc = reduce_from_scratch(D0);
    oineus::DecompositionManipStats st;
    dc.update_with_edits(new_to_old, newB, ndf, ndl, &st);

    REQUIRE(dc.r_data.size() == n_old + 2);
    REQUIRE(dc.sanity_check());
    REQUIRE(dc.d_data == newB);
    REQUIRE(dc.dim_first == ndf);
    REQUIRE(dc.dim_last == ndl);
    REQUIRE(pairing(dc) == pairing(reduce_from_scratch(newB)));
}

// ---------------------------------------------------------------------------
// Luo-Nelson Alg 3 : combined delete + reorder + insert
// ---------------------------------------------------------------------------
TEST_CASE("update_with_edits combined delete reorder insert")
{
    Decomp base = grid_decomp(4, 88);
    const MatrixData D0 = base.d_data;
    const size_t n_old = D0.size();
    const size_t top_dim = base.dim_first.size() - 1;
    const size_t top_hi = static_cast<size_t>(base.dim_last[top_dim]);
    const Int e0 = base.dim_first[1];

    // delete the last top cell; append a new top cell (boundary = three edges)
    std::vector<long long> new_to_old;
    for(size_t o = 0; o < n_old; ++o)
        if (o != top_hi)
            new_to_old.push_back(static_cast<long long>(o));
    MatrixData newB = build_survivor_boundary(D0, new_to_old);
    new_to_old.push_back(-1);
    newB.push_back(std::vector<Int>{e0, e0 + 2, e0 + 4});

    std::vector<int> dims;
    for(long long o : new_to_old)
        dims.push_back(o >= 0 ? cell_dim(static_cast<size_t>(o), base.dim_first, base.dim_last)
                              : static_cast<int>(top_dim));
    std::vector<Int> ndf, ndl;
    derive_dims(dims, ndf, ndl);

    Decomp dc = reduce_from_scratch(D0);
    oineus::DecompositionManipStats st;
    dc.update_with_edits(new_to_old, newB, ndf, ndl, &st);

    REQUIRE(dc.r_data.size() == new_to_old.size());
    REQUIRE(dc.sanity_check());
    REQUIRE(dc.d_data == newB);
    REQUIRE(dc.dim_first == ndf);
    REQUIRE(dc.dim_last == ndl);
    REQUIRE(pairing(dc) == pairing(reduce_from_scratch(newB)));
}
