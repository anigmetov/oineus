// Benchmark: Freudenthal grid filtration boundary/coboundary MATRIX CONSTRUCTION,
// the current Simplex-based filtration vs a real Filtration<FreudenthalCell> (the
// compact (anchor,type) cell + shared (co)boundary tables, Stage 2).
//
// Unlike bench_boundary.cpp -- which times a free-function prototype against the
// master -- this builds an ACTUAL Filtration<FreudenthalCell> and times its
// fil.boundary_matrix() / fil.coboundary_matrix(), so the measured speedup is the
// one the live reduction path sees. The compact filtration is built from the
// master's cells in the SAME order (presorted ctor), so the two boundary matrices
// must be column-for-column identical -- that equality is the correctness oracle.
//
//   ./bench_freudenthal_cell                 # D=2 and D=3
//   ./bench_freudenthal_cell --grid-side 48 --reps 7

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <oineus/oineus.h>

using Int = int;
using Real = double;
using Col = oineus::SparseColumn<Int>;
using MatrixData = std::vector<Col>;

template<class F>
static double time_ms(int reps, F&& f)
{
    std::vector<double> ts;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

static bool equal_mat(const MatrixData& a, const MatrixData& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t c = 0; c < a.size(); ++c) {
        if (a[c].size() != b[c].size())
            return false;
        for (size_t i = 0; i < a[c].size(); ++i)
            if (a[c][i] != b[c][i])
                return false;
    }
    return true;
}

static size_t nnz(const MatrixData& m)
{
    size_t s = 0;
    for (const auto& c : m)
        s += c.size();
    return s;
}

template<size_t D>
static void bench(size_t side, int reps)
{
    std::cout << "\n============ Freudenthal grid (D=" << D << ") ============\n";
    std::cout << "grid side=" << side << " top_d=" << D << "\n";

    using Grid = oineus::Grid<Int, Real, D>;
    using FrCell = oineus::FreudenthalCell<Int, D>;
    using FrFiltration = oineus::Filtration<FrCell, Real>;
    using FrCellValue = oineus::CellWithValue<FrCell, Real>;

    typename Grid::GridPoint dims;
    size_t total = 1;
    for (size_t d = 0; d < D; ++d) {
        dims[d] = static_cast<Int>(side);
        total *= side;
    }

    std::mt19937_64 gen(7);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::vector<Real> data(total);
    for (auto& x : data)
        x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);

    // master: the current Simplex-based Freudenthal filtration
    auto master = grid.freudenthal_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/1);
    const size_t n = master.size();
    std::cout << "filtration size (n_cols): " << n << "\n";

    using MasterCell = std::decay_t<decltype(master.cells()[0])>;
    std::cout << "per-cell footprint: master sizeof(cell)=" << sizeof(MasterCell)
              << " B + heap (dim+1)*" << sizeof(Int) << " B vertex array;"
              << " FreudenthalCell sizeof=" << sizeof(FrCellValue) << " B, no heap\n";

    // build the shared (anchor,type) tables once from the domain
    oineus::FrGeometry<Int, D> frgeom(grid.domain());
    std::cout << "n_types=" << frgeom.type_dim.size() << " type_bits=" << frgeom.type_bits << "\n";

    // compact cells, in the master's filtration order (so the boundary matrices line
    // up column-for-column for the equality oracle). Build via the presorted ctor.
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

    // ---- boundary ----
    MatrixData master_bd, fr_bd;
    double t_master_bd = time_ms(reps, [&]() { master_bd = master.boundary_matrix(1); });
    double t_fr_bd = time_ms(reps, [&]() { fr_bd = fr.boundary_matrix(1); });
    bool ok_bd = equal_mat(fr_bd, master_bd);

    // ---- coboundary ----
    MatrixData master_cob, fr_cob;
    double t_master_cob = time_ms(reps, [&]() { master_cob = master.coboundary_matrix(1); });
    double t_fr_cob = time_ms(reps, [&]() { fr_cob = fr.coboundary_matrix(1); });
    bool ok_cob = equal_mat(fr_cob, master_cob);

    std::cout << "boundary nnz=" << nnz(master_bd) << "  coboundary nnz=" << nnz(master_cob) << "\n";

    std::cout << "\nBOUNDARY build (ms, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (vector<vertex> Simplex)", t_master_bd);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "Filtration<FreudenthalCell>", t_fr_bd,
            t_fr_bd > 0 ? t_master_bd / t_fr_bd : 0.0, ok_bd ? "OK" : "MISMATCH");

    std::cout << "\nCOBOUNDARY build (ms, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (antitranspose)", t_master_cob);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "Filtration<FreudenthalCell> direct", t_fr_cob,
            t_fr_cob > 0 ? t_master_cob / t_fr_cob : 0.0, ok_cob ? "OK" : "MISMATCH");
}

int main(int argc, char** argv)
{
    size_t grid_side = 64;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if (a == "--grid-side") grid_side = std::stoul(need());
        else if (a == "--reps") reps = std::stoi(need());
        else { std::cerr << "unknown arg: " << a << "\n"; return 1; }
    }

    bench<2>(grid_side, reps);
    bench<3>(grid_side, reps);
    return 0;
}
