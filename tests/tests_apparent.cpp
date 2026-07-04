#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

#include <oineus/oineus.h>
#include <oineus/apparent.h>

using namespace oineus;

// SupportsApparent must accept exactly the cell families exposing BOTH buffer
// (co)boundary forms: slim Cube and slim Freudenthal. Fat Simplex (no coboundary)
// and BitPacked (packed boundary, no direct coboundary) stay out.
static_assert(SupportsApparent<Cube<int, 2>>::value);
static_assert(SupportsApparent<Cube<long, 3>>::value);
static_assert(SupportsApparent<Simplex<int, FreudenthalAnchorType<int, 2>>>::value);
static_assert(SupportsApparent<Simplex<long, FreudenthalAnchorType<long, 3>>>::value);
static_assert(not SupportsApparent<Simplex<int>>::value);
static_assert(not SupportsApparent<Simplex<long>>::value);
static_assert(not SupportsApparent<Simplex<int, BitPacked<int, std::uint64_t>>>::value);
static_assert(not SupportsApparent<Simplex<long, BitPacked<long, unsigned __int128>>>::value);

template<class AM>
static bool same_matching(const AM& a, const AM& b)
{
    return a.apparent_pivot_of_row == b.apparent_pivot_of_row
            and a.is_apparent_col == b.is_apparent_col
            and a.n_apparent == b.n_apparent;
}

TEST_CASE("apparent: generic detector == brute force on a hand-built complex")
{
    using Int = int;
    // Filled triangle on vertices {0,1,2}, sorted_id order:
    //   0,1,2 = vertices;  3=e(0,1), 4=e(0,2), 5=e(1,2);  6=t(0,1,2)
    // Boundary columns (sorted facet ids):
    std::vector<std::vector<Int>> D(7);
    D[3] = {0, 1};
    D[4] = {0, 2};
    D[5] = {1, 2};
    D[6] = {3, 4, 5};

    auto am_gen = detect_apparent_generic(D);
    auto am_bru = detect_apparent_bruteforce(D);

    REQUIRE(same_matching(am_gen, am_bru));

    // Hand-checked apparent pairs: edge 3=(0,1) is the youngest facet of nothing
    // until the triangle; low(D_6)=5, and column 6 is the first to contain row 5,
    // so (5,6) is apparent. (1,3): low(D_3)=1, col 3 is first to contain row 1 -> apparent.
    // (2,4): low(D_4)=2, col 4 is first to contain row 2 -> apparent.
    REQUIRE(am_gen.is_apparent(3));   // (1,3)
    REQUIRE(am_gen.is_apparent(4));   // (2,4)
    REQUIRE(am_gen.is_apparent(6));   // (5,6)
    REQUIRE(not am_gen.is_apparent(5)); // (1,2) is not the oldest cofacet of 1 or 2
    REQUIRE(am_gen.apparent_pivot_of_row[5] == 6);
    REQUIRE(am_gen.apparent_pivot_of_row[1] == 3);
    REQUIRE(am_gen.apparent_pivot_of_row[2] == 4);
}

TEST_CASE("apparent: generic == local == brute force on cubical, subset of persistence pairs")
{
    using Int = int;
    using Real = double;
    constexpr size_t D = 2;
    using Grid = oineus::Grid<Int, Real, D>;

    std::mt19937_64 gen(1234);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    typename Grid::GridPoint dims{7, 7};
    std::vector<Real> data(49);
    for(auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.cube_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);

    auto bd = fil.boundary_matrix(1);

    auto am_gen = detect_apparent_generic(bd);
    auto am_bru = detect_apparent_bruteforce(bd);
    auto am_loc = detect_apparent_local(fil);

    REQUIRE(same_matching(am_gen, am_bru));
    REQUIRE(same_matching(am_gen, am_loc));
    REQUIRE(am_gen.n_apparent > 0);

    // every apparent pair must be a true persistence pair of the full reduction
    VRUDecomposition<Int> dcmp(fil, /*dualize=*/false);
    ReductionParams p;
    p.compute_v = true;
    p.n_threads = 1;
    dcmp.reduce(p);

    for(size_t r = 0; r < fil.size(); ++r) {
        Int c = am_gen.apparent_pivot_of_row[r];
        if (c >= 0)
            REQUIRE(dcmp._pivots[r] == c);
    }
}

TEST_CASE("apparent: generic == local on 3D cubical")
{
    using Int = int;
    using Real = double;
    constexpr size_t D = 3;
    using Grid = oineus::Grid<Int, Real, D>;

    std::mt19937_64 gen(99);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    typename Grid::GridPoint dims{5, 5, 5};
    std::vector<Real> data(125);
    for(auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.cube_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);

    auto bd = fil.boundary_matrix(1);
    auto am_gen = detect_apparent_generic(bd);
    auto am_loc = detect_apparent_local(fil);

    REQUIRE(same_matching(am_gen, am_loc));
    REQUIRE(am_gen.n_apparent > 0);
}

TEST_CASE("apparent: generic == local == brute force on slim Freudenthal, subset of persistence pairs")
{
    using Int = int;
    using Real = double;
    constexpr size_t D = 2;
    using Grid = oineus::Grid<Int, Real, D>;

    std::mt19937_64 gen(4321);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    typename Grid::GridPoint dims{7, 7};
    std::vector<Real> data(49);
    for(auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.freudenthal_filtration_slim(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);

    auto bd = fil.boundary_matrix(1);

    auto am_gen = detect_apparent_generic(bd);
    auto am_bru = detect_apparent_bruteforce(bd);
    auto am_loc = detect_apparent_local(fil);

    REQUIRE(same_matching(am_gen, am_bru));
    REQUIRE(same_matching(am_gen, am_loc));
    REQUIRE(am_gen.n_apparent > 0);

    // every apparent pair must be a true persistence pair of the full reduction
    VRUDecomposition<Int> dcmp(fil, /*dualize=*/false);
    ReductionParams p;
    p.compute_v = true;
    p.n_threads = 1;
    dcmp.reduce(p);

    for(size_t r = 0; r < fil.size(); ++r) {
        Int c = am_gen.apparent_pivot_of_row[r];
        if (c >= 0)
            REQUIRE(dcmp._pivots[r] == c);
    }
}

TEST_CASE("apparent: R-only fused reduce matches classic pivots (cube + Freudenthal, hom + coh)")
{
    // compute_v=false path: no V exists, so R = D V cannot be checked; the
    // oracle is the pivot array (the pivot function of reduced R is canonical)
    // of a classic serial reduce of the same filtration
    using Int = int;
    using Real = double;
    constexpr size_t D = 3;
    using Grid = oineus::Grid<Int, Real, D>;

    std::mt19937_64 gen(2026);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    typename Grid::GridPoint dims{6, 5, 4};
    std::vector<Real> data(120);
    for(auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);

    auto check = [](const auto& fil) {
        for(bool dualize : {false, true}) {
            ReductionParams p_ref;
            p_ref.n_threads = 1;
            p_ref.compute_v = false;
            auto ref = VRUDecomposition<Int>::reduce_from_filtration_fused(fil, p_ref, dualize);
            REQUIRE(ref.n_apparent_pairs() == 0);

            ReductionParams p_app;
            p_app.n_threads = 4;
            p_app.compute_v = false;
            p_app.use_apparent_pairs = true;
            auto test = VRUDecomposition<Int>::reduce_from_filtration_fused(fil, p_app, dualize);

            REQUIRE(test.n_apparent_pairs() > 0);
            REQUIRE(test._pivots == ref._pivots);
        }
    };

    check(grid.cube_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1));
    check(grid.freudenthal_filtration_slim(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1));
}

TEST_CASE("apparent: generic == local on 3D slim Freudenthal, serial and parallel")
{
    using Int = int;
    using Real = double;
    constexpr size_t D = 3;
    using Grid = oineus::Grid<Int, Real, D>;

    std::mt19937_64 gen(777);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    typename Grid::GridPoint dims{5, 5, 5};
    std::vector<Real> data(125);
    for(auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.freudenthal_filtration_slim(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);

    auto bd = fil.boundary_matrix(1);
    auto am_gen = detect_apparent_generic(bd);
    auto am_loc = detect_apparent_local(fil);
    auto am_par = detect_apparent_local(fil, /*n_threads=*/4);

    REQUIRE(same_matching(am_gen, am_loc));
    REQUIRE(same_matching(am_gen, am_par));
    REQUIRE(am_gen.n_apparent > 0);
}
