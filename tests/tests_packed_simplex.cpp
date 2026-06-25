#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
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
