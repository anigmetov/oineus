#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <oineus/oineus.h>

using namespace oineus;

namespace {

using Int = int;
using Real = double;
using Word = std::uint64_t;
using Packed = oineus::BitPacked<Int, Word>;
using PackedCell = oineus::Simplex<Int, Packed>;
using FatSimplex = oineus::Simplex<Int>;
using Uid128 = typename FatSimplex::Uid;

// ---- E.4 SFINAE gate (compile-time contract) ----
// get_vertices / no-arg boundary() / join are valid only for the self-contained Fat
// encoding; the geometry-bearing encodings (BitPacked, FreudenthalAnchorType) must NOT
// expose them (they use vertices(geom) / boundary_into instead). This is the only place
// the gate's contract can be locked in: from Python every materialized cell is fat, so a
// relaxed enable_if that silently re-exposed a broken no-arg boundary() on a slim cell
// would slip past the Python tests. The static_asserts below fail the build if the gate
// regresses.
template<class, class = void> struct has_get_vertices : std::false_type {};
template<class T> struct has_get_vertices<T, std::void_t<decltype(std::declval<const T&>().get_vertices())>> : std::true_type {};

template<class, class = void> struct has_no_arg_boundary : std::false_type {};
template<class T> struct has_no_arg_boundary<T, std::void_t<decltype(std::declval<const T&>().boundary())>> : std::true_type {};

template<class, class = void> struct has_join : std::false_type {};
template<class T> struct has_join<T, std::void_t<decltype(std::declval<const T&>().join(std::declval<typename T::Int>(), std::declval<typename T::Int>()))>> : std::true_type {};

using FrCell2 = oineus::Simplex<Int, oineus::FreudenthalAnchorType<Int, 2>>;

static_assert(has_get_vertices<FatSimplex>::value, "fat Simplex must expose get_vertices()");
static_assert(!has_get_vertices<PackedCell>::value, "bit-packed cell must NOT expose get_vertices()");
static_assert(!has_get_vertices<FrCell2>::value, "Freudenthal cell must NOT expose get_vertices()");

static_assert(has_no_arg_boundary<FatSimplex>::value, "fat Simplex must expose no-arg boundary()");
static_assert(!has_no_arg_boundary<PackedCell>::value, "bit-packed cell must NOT expose no-arg boundary()");
static_assert(!has_no_arg_boundary<FrCell2>::value, "Freudenthal cell must NOT expose no-arg boundary()");

static_assert(has_join<FatSimplex>::value, "fat Simplex must expose join()");
static_assert(!has_join<PackedCell>::value, "bit-packed cell must NOT expose join()");
static_assert(!has_join<FrCell2>::value, "Freudenthal cell must NOT expose join()");

// fat 128-bit combinatorial uid of a vertex set (the master's uid space), used to
// compare the slim packed (co)boundary against the fat one in vertex-set space
Uid128 fat_uid(const std::vector<Int>& v)
{
    typename FatSimplex::IdxVector iv(v.begin(), v.end());
    return FatSimplex(iv).get_uid();
}

// collect the wrapper's alloc-elided buffer boundary into a vector of packed words
std::vector<Word> bd_words(const PackedCell& c, const PackedGeom& g)
{
    std::vector<Word> r;
    c.boundary_into(g, [&r](Word u) { r.push_back(u); });
    return r;
}

// The packed VR builder (get_vr_filtration_packed_inorder<...,W>) must produce a
// filtration cell-for-cell identical to the fat builder (same enumeration order, same
// values, hence identical (co)boundary matrices and diagrams), with the critical-edge
// array aligned, and each packed cell materializing back to the fat vertex set. Run as
// a templated helper so it covers both word widths (u64 and __int128); the caller
// guarantees the points/max_dim force the requested width via bit_packing_fits.
template<class W, std::size_t D>
void check_packed_vr_builder(const std::vector<oineus::Point<Real, D>>& points,
                             dim_type max_dim, Real max_diameter, int n_threads = 1)
{
    using Pk = oineus::BitPacked<Int, W>;
    using PkCell = oineus::Simplex<Int, Pk>;
    const size_t n_points = points.size();
    REQUIRE(oineus::bit_packing_fits<W>(n_points, max_dim));

    // compare against the FAT VRE builder (same enumeration order), not the
    // Bron-Kerbosch get_vr_filtration, so the cell orders -- hence the (co)boundary
    // matrices -- coincide. n_threads > 1 also exercises the parallel add_cell sink.
    auto fat = oineus::get_vr_filtration_inorder<Int, Real, D>(points, max_dim, max_diameter, n_threads);
    auto packed = oineus::get_vr_filtration_packed_inorder<Int, Real, W, D>(points, max_dim, max_diameter, n_threads);

    REQUIRE(packed.size() == fat.size());
    REQUIRE(packed.size() > 0);

    // identical enumeration order + values -> column-identical (co)boundary matrices
    REQUIRE(packed.boundary_matrix(1) == fat.boundary_matrix(1));
    REQUIRE(packed.coboundary_matrix(1) == fat.coboundary_matrix(1));

    // the packed filtration reduces to a valid R = D V (exercises the packed-boundary +
    // hash-uid-index + antitranspose-coboundary path, newly for the __int128 hash)
    VRUDecomposition<Int> dcmp(packed, /*dualize=*/false);
    Params p;
    p.compute_v = true;
    p.n_threads = 1;
    dcmp.reduce(p);
    REQUIRE(dcmp.sanity_check());

    // per-cell: same value, and the packed cell materializes to the fat vertex set
    const auto& geom = packed.geometry();
    for (size_t c = 0; c < packed.size(); ++c) {
        REQUIRE(packed.cells()[c].get_value() == fat.cells()[c].get_value());
        const PkCell& cell = packed.cells()[c].get_cell();
        std::vector<Int> fat_vs(fat.cells()[c].get_cell().get_vertices().begin(),
                                fat.cells()[c].get_cell().get_vertices().end());
        REQUIRE(cell.vertices(geom) == fat_vs);
    }

    // critical edges: same array, aligned to the (presorted) cells. Compare as
    // unordered endpoint pairs since VREdge stores {s, t} (s may be > t).
    auto [fat_e, fat_edges] = oineus::get_vr_filtration_and_critical_edges_inorder<Int, Real, D>(
            points, max_dim, max_diameter, n_threads);
    auto [packed_e, packed_edges] = oineus::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, W, D>(
            points, max_dim, max_diameter, n_threads);
    REQUIRE(packed_edges.size() == fat_edges.size());
    REQUIRE(packed_edges.size() == packed.size());
    // both builders run the identical VRE enumeration, so the per-cell VREdge is
    // byte-identical (same endpoint order), not merely the same unordered pair
    for (size_t i = 0; i < fat_edges.size(); ++i)
        REQUIRE(fat_edges[i] == packed_edges[i]);
}

} // namespace

TEST_CASE("bit-packed simplex: pack / unpack / boundary by hand")
{
    const int bits = 3;
    PackedGeom g{bits};

    // 2-simplex {1,2,5}, ascending, 3-bit fields: 5<<6 | 2<<3 | 1 = 0x151
    std::vector<Int> vs{1, 2, 5};
    Packed enc(vs, bits);
    REQUIRE(enc.dim() == 2);
    REQUIRE(enc.get_uid() == ((Word(5) << 6) | (Word(2) << 3) | Word(1)));
    REQUIRE(enc.vertices(g) == vs);

    // boundary: drop each vertex, repack the remaining two contiguously
    std::vector<Word> bd = enc.boundary(g);
    REQUIRE(bd.size() == 3);
    // facets {2,5}, {1,5}, {1,2} (dropping 1, 2, 5 in field order)
    Packed f0(std::vector<Int>{2, 5}, bits);
    Packed f1(std::vector<Int>{1, 5}, bits);
    Packed f2(std::vector<Int>{1, 2}, bits);
    REQUIRE(bd[0] == f0.get_uid());
    REQUIRE(bd[1] == f1.get_uid());
    REQUIRE(bd[2] == f2.get_uid());

    // a vertex has empty boundary
    Packed v(std::vector<Int>{4}, bits);
    REQUIRE(v.dim() == 0);
    REQUIRE(v.boundary(g).empty());
    REQUIRE(v.vertices(g) == std::vector<Int>{4});
}

// vertices_from_simplex_uid is the inverse of simplex_uid: it recovers the ascending
// vertex set from the universal combinatorial uid. It is the Python-facing uid-contract
// translation -- a (fat) simplex's uid decodes back to vertices, which the slim/packed
// encodings re-key into their own internal uid. Property-test the round-trip across
// dimensions, including vertex 0 and the all-minimal simplex (uid body 0).
TEST_CASE("combinatorial uid round-trip: vertices_from_simplex_uid(simplex_uid(v)) == v")
{
    auto check = [](std::vector<Int> v) {
        std::sort(v.begin(), v.end());
        Uid128 uid = simplex_uid<Int>(v);
        std::vector<Int> back = vertices_from_simplex_uid<Int>(uid);
        REQUIRE(back == v);
    };

    // explicit edge cases
    check({0});                 // vertex 0: uid body 0
    check({7});                 // a single non-zero vertex
    check({0, 1});              // lowest edge: uid body 0
    check({0, 1, 2, 3, 4});     // lowest 4-simplex
    check({3, 8});
    check({2, 5, 9});
    check({10, 11, 12, 13});
    check({0, 100, 5000});      // wide vertex spread

    // exhaustive small simplices: every strictly-increasing vertex set drawn from
    // {0..15} of size 1..4
    for (int n = 0; n <= 15; ++n)
        for (int a = n + 1; a <= 15; ++a) {
            check({n});
            check({n, a});
            for (int b = a + 1; b <= 15; ++b) {
                check({n, a, b});
                for (int c = b + 1; c <= 15; ++c)
                    check({n, a, b, c});
            }
        }

    // a garbage uid whose decoded vertex id exceeds the Int range must throw out_of_range
    // (the binding layer maps this to "uid not present"), never overflow / return garbage.
    using Uid = Uid128;
    // single vertex with id 2^31 > INT_MAX: dim_info = (1+1)<<124, body = comb(2^31,1) = 2^31
    Uid uid_big_vertex = (Uid(2) << 124) | Uid(2147483648ULL);
    REQUIRE_THROWS_AS(vertices_from_simplex_uid<Int>(uid_big_vertex), std::out_of_range);
    // a 3-vertex uid with a near-maximal 124-bit body decodes to vertices far beyond INT_MAX
    Uid uid_huge_triangle = (Uid(4) << 124) | ((Uid(1) << 124) - 1);
    REQUIRE_THROWS_AS(vertices_from_simplex_uid<Int>(uid_huge_triangle), std::out_of_range);
}

// Build a Filtration<Simplex<...,BitPacked>> from a fat Vietoris-Rips master, in the
// SAME order, and verify (a) the boundary/coboundary matrices are column-identical to
// the fat master, (b) it reduces to a valid R = D V, and (c) per cell the materializer
// round-trips and the slim boundary matches the fat Simplex boundary in vertex-set
// space (the geometric ground truth, since the packed uid and the fat combinatorial
// uid live in different spaces).
TEST_CASE("bit-packed simplex: filtration matches fat VR master + materialization")
{
    constexpr std::size_t D = 3;
    const std::vector<oineus::Point<Real, D>> points = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1},
    };
    const size_t n_points = points.size();
    const dim_type max_dim = 3;

    auto master = oineus::get_vr_filtration<Int, Real, D>(points, max_dim,
            std::numeric_limits<Real>::max(), /*n_threads=*/1);
    const size_t n = master.size();
    REQUIRE(n > 0);

    const int bits = oineus::packed_vertex_bits(n_points);
    REQUIRE(oineus::bit_packing_fits<Word>(n_points, max_dim));
    PackedGeom geom{bits};

    using PackedFiltration = oineus::Filtration<PackedCell, Real>;
    typename PackedFiltration::CellVector cells;
    cells.reserve(n);
    for (size_t c = 0; c < n; ++c) {
        const auto& mc = master.cells()[c];
        std::vector<Int> vids(mc.get_cell().get_vertices().begin(),
                              mc.get_cell().get_vertices().end());
        cells.emplace_back(PackedCell(Packed(vids, bits)), mc.get_value());
    }
    PackedFiltration fil(oineus::presorted, std::move(cells), /*negate=*/false);
    fil.set_geometry(geom);

    // (a) column-identical matrices (packed buffer boundary + hash uid index, and the
    // antitranspose coboundary -- BitPacked is HasDirectCoboundary == false)
    REQUIRE(fil.boundary_matrix(1) == master.boundary_matrix(1));
    REQUIRE(fil.coboundary_matrix(1) == master.coboundary_matrix(1));

    // (b) reduces to a valid decomposition
    VRUDecomposition<Int> dcmp(fil, /*dualize=*/false);
    Params p;
    p.compute_v = true;
    p.n_threads = 1;
    dcmp.reduce(p);
    REQUIRE(dcmp.sanity_check());

    // (c) per-cell materialization vs the fat Simplex
    for (size_t c = 0; c < n; ++c) {
        const auto& sigma = master.cells()[c].get_cell();
        std::vector<Int> vids(sigma.get_vertices().begin(), sigma.get_vertices().end());
        // cells()[c] returns a reference; get_cell(c) returns BY VALUE, so a reference
        // bound through it would dangle
        const PackedCell& cell = fil.cells()[c].get_cell();

        // the materializer round-trips the original vertex set
        REQUIRE(cell.vertices(geom) == vids);

        // slim boundary == fat boundary, as multisets of fat 128-bit uids
        std::vector<Uid128> b_slim;
        const dim_type facet_dim = cell.dim() == 0 ? 0 : cell.dim() - 1;
        for (Word fw : bd_words(cell, geom)) {
            Packed facet(fw, facet_dim);
            b_slim.push_back(fat_uid(facet.vertices(geom)));
        }
        std::vector<Uid128> b_fat = sigma.boundary();
        std::sort(b_slim.begin(), b_slim.end());
        std::sort(b_fat.begin(), b_fat.end());
        REQUIRE(b_slim == b_fat);
    }
}

TEST_CASE("bit-packed VR builder: packed (uint64) matches fat builder, 3D cloud")
{
    constexpr std::size_t D = 3;
    const std::vector<oineus::Point<Real, D>> points = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1},
    };
    // n_points = 7 -> bits = 3; a 3-simplex needs 4*3 = 12 bits, fits uint64
    REQUIRE(oineus::bit_packing_fits<std::uint64_t>(points.size(), 3));
    for (int nt : {1, 4})
        check_packed_vr_builder<std::uint64_t, D>(points, /*max_dim=*/3, std::numeric_limits<Real>::max(), nt);
}

TEST_CASE("bit-packed VR builder: packed (__int128) matches fat builder, tiered overflow")
{
    using Word128 = unsigned __int128;
    constexpr std::size_t D = 3;
    // A tight 6-point cluster (forms a full 5-simplex) plus many isolated far points,
    // so n_points is large (bits = 11) while the complex stays tiny. A 5-simplex needs
    // 6*11 = 66 bits: too wide for uint64 (forces the __int128 tier), fits __int128.
    std::vector<oineus::Point<Real, D>> points;
    const double e = 0.01;
    points.push_back({0, 0, 0});
    points.push_back({e, 0, 0});
    points.push_back({0, e, 0});
    points.push_back({0, 0, e});
    points.push_back({e, e, 0});
    points.push_back({e, 0, e});
    // isolated far points: 100 apart on the x-axis, none within max_diameter of anything
    for (int m = 0; m < 1094; ++m)
        points.push_back({100.0 * (m + 1), 0, 0});

    const size_t n_points = points.size();   // 1100
    const dim_type max_dim = 5;
    REQUIRE(oineus::packed_vertex_bits(n_points) == 11);
    REQUIRE_FALSE(oineus::bit_packing_fits<std::uint64_t>(n_points, max_dim));
    REQUIRE(oineus::bit_packing_fits<Word128>(n_points, max_dim));

    for (int nt : {1, 4})
        check_packed_vr_builder<Word128, D>(points, max_dim, /*max_diameter=*/0.5, nt);
}

TEST_CASE("E.4 SFINAE gate: Fat-only methods absent on geometry-bearing encodings")
{
    // mirrors the file-scope static_asserts (which already fail the build on regression);
    // surfaced here so the contract shows up as a ctest case
    CHECK(has_get_vertices<FatSimplex>::value);
    CHECK_FALSE(has_get_vertices<PackedCell>::value);
    CHECK_FALSE(has_get_vertices<FrCell2>::value);

    CHECK(has_no_arg_boundary<FatSimplex>::value);
    CHECK_FALSE(has_no_arg_boundary<PackedCell>::value);
    CHECK_FALSE(has_no_arg_boundary<FrCell2>::value);

    CHECK(has_join<FatSimplex>::value);
    CHECK_FALSE(has_join<PackedCell>::value);
    CHECK_FALSE(has_join<FrCell2>::value);
}
