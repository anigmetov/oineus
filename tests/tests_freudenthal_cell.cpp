#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <oineus/oineus.h>

using namespace oineus;

// Build a Filtration<FreudenthalCell> from a grid's Simplex Freudenthal filtration,
// in the SAME order (presorted ctor), so the two boundary matrices must be
// column-for-column identical -- a tight oracle for the compact (anchor,type) cell
// and its table-driven (co)boundary.
template<size_t D>
static void check_freudenthal_cell(const std::array<int, D>& dims)
{
    using Int = int;
    using Real = double;
    using Grid = oineus::Grid<Int, Real, D>;
    using FrCell = oineus::FreudenthalCell<Int, D>;
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
        cells.emplace_back(FrCell(uid, frgeom.dim_of_uid(uid)), mc.get_value());
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
