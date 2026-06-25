#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <oineus/oineus.h>

using namespace oineus;

// The Simplex<Int,Enc> wrapper exposes the alloc-elided boundary_into / coboundary_into
// (the forms the Filtration's packed builders use); collect them into a vector for the
// per-cell geometric checks below.
template<class Cell, class Geom>
static std::vector<typename Cell::Int> bd_uids(const Cell& c, const Geom& g)
{
    std::vector<typename Cell::Int> r;
    c.boundary_into(g, [&r](typename Cell::Int u) { r.push_back(u); });
    return r;
}

template<class Cell, class Geom>
static std::vector<typename Cell::Int> cob_uids(const Cell& c, const Geom& g)
{
    std::vector<typename Cell::Int> r;
    c.coboundary_into(g, [&r](typename Cell::Int u) { r.push_back(u); });
    return r;
}

// Build a Filtration<Simplex<...,FreudenthalAnchorType>> from a grid's Simplex
// Freudenthal filtration, in the SAME order (presorted ctor), so the two boundary
// matrices must be column-for-column identical -- a tight oracle for the compact
// (anchor,type) cell and its table-driven (co)boundary.
template<size_t D>
static void check_freudenthal_cell(const std::array<int, D>& dims)
{
    using Int = int;
    using Real = double;
    using Grid = oineus::Grid<Int, Real, D>;
    using FrEnc = oineus::FreudenthalAnchorType<Int, D>;
    using FrCell = oineus::Simplex<Int, FrEnc>;
    using FrFiltration = oineus::Filtration<FrCell, Real>;

    typename Grid::GridPoint gp;
    size_t total = 1;
    for (size_t d = 0; d < D; ++d) { gp[d] = dims[d]; total *= dims[d]; }

    std::mt19937_64 gen(1234);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::vector<Real> data(total);
    for (auto& x : data) x = dist(gen);

    Grid grid(gp, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto master = grid.freudenthal_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);
    const size_t n = master.size();
    REQUIRE(n > 0);

    oineus::FrGeometry<Int, D> frgeom(grid.domain());

    typename FrFiltration::CellVector cells;
    cells.reserve(n);
    for (size_t c = 0; c < n; ++c) {
        const auto& mc = master.cells()[c];
        const auto& vs = mc.get_cell().get_vertices();
        std::vector<Int> vids(vs.begin(), vs.end());
        Int uid = frgeom.uid_of_vertices(vids);
        cells.emplace_back(FrCell(FrEnc(uid, frgeom.dim_of_uid(uid))), mc.get_value());
    }
    FrFiltration fr(oineus::presorted, std::move(cells), /*negate=*/false);
    fr.set_geometry(frgeom);

    // the compact cell's table-driven (co)boundary must reproduce the Simplex one
    REQUIRE(fr.boundary_matrix(1) == master.boundary_matrix(1));
    REQUIRE(fr.coboundary_matrix(1) == master.coboundary_matrix(1));

    // the new cell type reduces to a valid decomposition (R = D V)
    VRUDecomposition<Int> dcmp(fr, /*dualize=*/false);
    Params p;
    p.compute_v = true;
    p.n_threads = 1;
    dcmp.reduce(p);
    REQUIRE(dcmp.sanity_check());
}

TEST_CASE("freudenthal cell: compact (anchor,type) cell matches Simplex, 2D")
{
    check_freudenthal_cell<2>({9, 7});
}

TEST_CASE("freudenthal cell: compact (anchor,type) cell matches Simplex, 3D")
{
    check_freudenthal_cell<3>({6, 5, 7});
}

// Geometric ground-truth check for the slim (anchor,type) cell: materialize each
// cell back to its real grid vertex ids (FrGeometry::vertices_of, the slim->fat
// Freudenthal materializer) and verify, IN VERTEX-SET SPACE, that
//   - the materializer round-trips uid_of_vertices,
//   - the slim boundary equals the boundary of the fat Simplex on those vertices,
//   - the slim coboundary equals the FULL-triangulation coboundary, obtained
//     independently by inverting the fat boundary of every simplex (the master with
//     top_d=D is the complete Freudenthal triangulation).
// This compares fat 128-bit combinatorial uids (which live in a different space than
// the slim anchor<<type_bits|type uids), so it is an independent geometric oracle,
// not a slim-vs-slim restatement of check_freudenthal_cell above.
template<size_t D>
static void check_freudenthal_materialization(const std::array<int, D>& dims)
{
    using Int = int;
    using Real = double;
    using Grid = oineus::Grid<Int, Real, D>;
    using FrEnc = oineus::FreudenthalAnchorType<Int, D>;
    using FrCell = oineus::Simplex<Int, FrEnc>;
    using Simplex = oineus::Simplex<Int>;
    using Uid128 = typename Simplex::Uid;

    typename Grid::GridPoint gp;
    size_t total = 1;
    for (size_t d = 0; d < D; ++d) { gp[d] = dims[d]; total *= dims[d]; }

    std::mt19937_64 gen(987);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::vector<Real> data(total);
    for (auto& x : data) x = dist(gen);

    Grid grid(gp, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto master = grid.freudenthal_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);
    const size_t n = master.size();
    REQUIRE(n > 0);

    oineus::FrGeometry<Int, D> frgeom(grid.domain());

    // fat 128-bit uid of a vertex set (allocator-converting into the Simplex IdxVector)
    auto fat_uid = [](const std::vector<Int>& v) {
        typename Simplex::IdxVector iv(v.begin(), v.end());
        return Simplex(iv).get_uid();
    };

    // independent coboundary oracle: parent simplices of each facet, keyed by fat uid
    std::map<Uid128, std::set<Uid128>> cob_oracle;
    for (size_t c = 0; c < n; ++c) {
        const auto& sigma = master.cells()[c].get_cell();
        Uid128 su = sigma.get_uid();
        for (Uid128 f : sigma.boundary())
            cob_oracle[f].insert(su);
    }

    for (size_t c = 0; c < n; ++c) {
        const auto& sigma = master.cells()[c].get_cell();
        std::vector<Int> vids(sigma.get_vertices().begin(), sigma.get_vertices().end());
        // master cells are already sorted, but be explicit -- vertices_of returns sorted
        std::sort(vids.begin(), vids.end());

        Int uid = frgeom.uid_of_vertices(vids);
        FrCell cell(FrEnc(uid, frgeom.dim_of_uid(uid)));

        // (1) materializer round-trips the encoding
        REQUIRE(frgeom.vertices_of(uid) == vids);
        REQUIRE(cell.vertices(frgeom) == vids);

        // (2) slim boundary == fat boundary, as multisets of fat uids
        std::vector<Uid128> b_slim;
        for (Int fu : bd_uids(cell, frgeom))
            b_slim.push_back(fat_uid(frgeom.vertices_of(fu)));
        std::vector<Uid128> b_fat = sigma.boundary();
        std::sort(b_slim.begin(), b_slim.end());
        std::sort(b_fat.begin(), b_fat.end());
        REQUIRE(b_slim == b_fat);

        // (3) slim coboundary == inverted-triangulation coboundary
        std::set<Uid128> c_slim;
        for (Int cu : cob_uids(cell, frgeom))
            c_slim.insert(fat_uid(frgeom.vertices_of(cu)));
        auto it = cob_oracle.find(sigma.get_uid());
        std::set<Uid128> c_oracle = (it == cob_oracle.end()) ? std::set<Uid128>{} : it->second;
        REQUIRE(c_slim == c_oracle);
    }
}

TEST_CASE("freudenthal cell: slim (co)boundary matches materialized fat simplex, 2D")
{
    check_freudenthal_materialization<2>({9, 7});
}

TEST_CASE("freudenthal cell: slim (co)boundary matches materialized fat simplex, 3D")
{
    check_freudenthal_materialization<3>({6, 5, 7});
}

// Hand-worked smallest case: one unit square Kuhn-split into two triangles. The
// counts and the diagonal edge's (co)boundary are derived by hand and asserted on
// the materialized slim cells, independent of which diagonal the triangulation picks.
TEST_CASE("freudenthal cell: 2x2 hand-checked triangulation, 2D")
{
    using Int = int;
    using Real = double;
    using Grid = oineus::Grid<Int, Real, 2>;
    using FrEnc = oineus::FreudenthalAnchorType<Int, 2>;
    using FrCell = oineus::Simplex<Int, FrEnc>;

    typename Grid::GridPoint gp{2, 2};
    std::vector<Real> data(4, 0.0);
    Grid grid(gp, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto master = grid.freudenthal_filtration(/*top_d=*/2, /*negate=*/false, /*n_threads=*/1);
    oineus::FrGeometry<Int, 2> frgeom(grid.domain());

    auto slim_of = [&](size_t c) {
        std::vector<Int> vs(master.cells()[c].get_cell().get_vertices().begin(),
                            master.cells()[c].get_cell().get_vertices().end());
        Int uid = frgeom.uid_of_vertices(vs);
        return std::make_pair(FrCell(FrEnc(uid, frgeom.dim_of_uid(uid))), vs);
    };
    auto point_of = [&](Int id) { return grid.domain().id_to_point(id); };

    // one square Kuhn-split -> 4 vertices, 5 edges (4 axis + 1 diagonal), 2 triangles
    std::array<int, 3> ndim{0, 0, 0};
    for (size_t c = 0; c < master.size(); ++c)
        ndim[master.cells()[c].get_cell().dim()]++;
    REQUIRE(ndim[0] == 4);
    REQUIRE(ndim[1] == 5);
    REQUIRE(ndim[2] == 2);

    int n_diag = 0;
    for (size_t c = 0; c < master.size(); ++c) {
        if (master.cells()[c].get_cell().dim() != 1)
            continue;
        auto [cell, vs] = slim_of(c);
        auto p0 = point_of(vs[0]);
        auto p1 = point_of(vs[1]);
        bool diagonal = (p0[0] != p1[0]) and (p0[1] != p1[1]);

        if (not diagonal) {
            // an axis-aligned edge bounds exactly one of the two triangles
            REQUIRE(cob_uids(cell, frgeom).size() == 1);
            continue;
        }
        ++n_diag;

        // the diagonal's boundary is exactly its two endpoints
        std::set<Int> got_b;
        for (Int fu : bd_uids(cell, frgeom)) {
            auto fv = frgeom.vertices_of(fu);
            REQUIRE(fv.size() == 1);
            got_b.insert(fv[0]);
        }
        REQUIRE(got_b == std::set<Int>(vs.begin(), vs.end()));

        // the diagonal is shared by both triangles, each containing both its endpoints
        auto cob = cob_uids(cell, frgeom);
        REQUIRE(cob.size() == 2);
        for (Int cu : cob) {
            auto cv = frgeom.vertices_of(cu);
            REQUIRE(cv.size() == 3);
            for (Int v : vs)
                REQUIRE(std::find(cv.begin(), cv.end(), v) != cv.end());
        }
    }
    REQUIRE(n_diag == 1);
}
