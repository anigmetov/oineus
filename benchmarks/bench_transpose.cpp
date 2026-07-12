// Times the col->row transpose strategies (column scatter, row scatter,
// row-bucket radix) on the V and U matrices of real reductions, replicating
// the two production call patterns:
//   - full-matrix V transpose (make_dynamic / ensure_ri_v_, the critical-set
//     path), and
//   - per-dimension U transpose (compute_u / compute_u_from_v).
//
// The strategy choice is architecture-sensitive (the transpose literature
// reports large CPU-dependent differences), so this benchmark exists to
// re-characterize the routing on new machines.
//
// Usage:
//   bench_transpose fr   N        MODE [n_threads_csv] [reps]   # N^3 grid
//   bench_transpose vr   N_POINTS MODE [n_threads_csv] [reps]   # full VR, max_dim=2
//   bench_transpose load FILE     MODE [n_threads_csv] [reps]   # dumped filtration
//
// MODE in {col, row}: force-routes col_to_row_format_parallel via
// OINEUS_TRANSPOSE_MODE (the env is latched on first use, hence one process
// per mode); the bucket variant is timed in the same process via a direct
// call. Both homology and cohomology reductions are benchmarked.
//
// load FILE format (little-endian): int64 n_dims, then per dim:
// int64 n, int64 k(=dim+1), n*k int64 vertex ids, n float64 values.
//
// Output: CSV lines  matrix,side,mode,dim,threads,seconds  on stdout
// (dim = -1 for the full-matrix V transpose).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <oineus/oineus.h>

using Int = long;
using Real = double;
using Simp = oineus::Simplex<Int>;
using Fil = oineus::Filtration<Simp, Real>;
using ST = oineus::SimpleSparseMatrixTraits<Int, 2>;
using MatrixData = typename ST::Matrix;

static double median3(double a, double b, double c)
{
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// one timed call of the routed production transpose (mode picked by env)
static double time_routed(const MatrixData& m, int threads, size_t cs, size_t ce, Int n_rows)
{
    Timer t;
    auto r = ST::col_to_row_format_parallel(m, threads, cs, ce, n_rows);
    double el = t.elapsed();
    (void)r;
    return el;
}

static double time_bucket(const MatrixData& m, int threads, size_t cs, size_t ce, Int n_rows)
{
    Timer t;
    auto r = ST::col_to_row_format_bucket(m, threads, cs, ce, n_rows);
    double el = t.elapsed();
    (void)r;
    return el;
}

template<class F>
static double median_of(F&& f, int reps)
{
    if (reps >= 3)
        return median3(f(), f(), f());
    double best = f();
    for (int i = 1; i < reps; ++i)
        best = std::min(best, f());
    return best;
}

static void bench_matrix(const char* matrix_name, const char* side, const char* mode,
                         const MatrixData& m, Int n_rows,
                         const std::vector<std::pair<int, std::pair<size_t, size_t>>>& jobs,
                         const std::vector<int>& thread_counts, int reps)
{
    for (const auto& [dim, range] : jobs) {
        size_t nnz = 0;
        for (size_t c = range.first; c < range.second; ++c)
            nnz += m[c].size();
        std::fprintf(stderr, "# %s %s dim=%d cols=%zu rows=%ld nnz=%zu nnz/row=%.2f\n",
                matrix_name, side, dim, range.second - range.first,
                static_cast<long>(n_rows), nnz, double(nnz) / double(n_rows));
        for (int threads : thread_counts) {
            const double t_routed = median_of(
                    [&] { return time_routed(m, threads, range.first, range.second, n_rows); }, reps);
            std::printf("%s,%s,%s,%d,%d,%.6f\n", matrix_name, side, mode, dim, threads, t_routed);
            const double t_bucket = median_of(
                    [&] { return time_bucket(m, threads, range.first, range.second, n_rows); }, reps);
            std::printf("%s,%s,bucket,%d,%d,%.6f\n", matrix_name, side, dim, threads, t_bucket);
            std::fflush(stdout);
        }
    }
}

// Anti-transpose via the bucket transpose plus a parallel per-column
// reverse-complement remap: antitranspose(a)[j] = { n-1-k : k in
// transpose(a)[n-1-j] } in ascending order. The remap is one extra
// sequential O(nnz) pass, so this is an UPPER bound on a native
// reverse-indexed bucket anti-transpose.
static MatrixData antitranspose_bucket_emulated(const MatrixData& a, size_t n, int threads)
{
    auto t = ST::col_to_row_format_bucket(a, threads, 0, a.size(), static_cast<Int>(n));
    t.resize(n);
    MatrixData result(n);
    const size_t nw = threads > 0 ? static_cast<size_t>(threads) : 1;
    std::vector<std::thread> ws;
    ws.reserve(nw);
    for (size_t w = 0; w < nw; ++w) {
        ws.emplace_back([&, w]() {
            for (size_t j = w; j < n; j += nw) {
                const auto& src = t[n - 1 - j];
                auto& dst = result[j];
                dst.resize(src.size());
                for (size_t p = 0; p < src.size(); ++p)
                    dst[p] = static_cast<Int>(n - 1 - static_cast<size_t>(src[src.size() - 1 - p]));
            }
        });
    }
    for (auto& t_ : ws) t_.join();
    return result;
}

static void bench_antitranspose(const std::string& matrix_name, const MatrixData& bdry,
                                size_t n, const std::vector<int>& thread_counts, int reps)
{
    size_t nnz = 0;
    for (const auto& col : bdry)
        nnz += col.size();
    std::fprintf(stderr, "# %s_D antitranspose cols=%zu nnz=%zu\n", matrix_name.c_str(), n, nnz);

    // one-time correctness check of the emulation against production
    {
        auto prod = oineus::antitranspose(bdry, n, 8);
        auto emu = antitranspose_bucket_emulated(bdry, n, 8);
        if (prod != emu) {
            std::fprintf(stderr, "FATAL: bucket-emulated antitranspose differs from production\n");
            std::exit(2);
        }
    }
    for (int threads : thread_counts) {
        const double t_prod = median_of([&] {
            Timer t;
            auto r = oineus::antitranspose(bdry, n, threads);
            double el = t.elapsed();
            (void)r;
            return el;
        }, reps);
        std::printf("%s_D,both,antitrans,-1,%d,%.6f\n", matrix_name.c_str(), threads, t_prod);
        const double t_emu = median_of([&] {
            Timer t;
            auto r = antitranspose_bucket_emulated(bdry, n, threads);
            double el = t.elapsed();
            (void)r;
            return el;
        }, reps);
        std::printf("%s_D,both,antitrans_bucket,-1,%d,%.6f\n", matrix_name.c_str(), threads, t_emu);
        std::fflush(stdout);
    }
}

static Fil load_filtration(const char* path)
{
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    auto rd = [&](void* p, size_t n) {
        if (std::fread(p, 1, n, f) != n) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    };
    std::int64_t n_dims;
    rd(&n_dims, 8);
    typename Fil::CellVector cells;
    for (std::int64_t d = 0; d < n_dims; ++d) {
        std::int64_t n, k;
        rd(&n, 8);
        rd(&k, 8);
        std::vector<std::int64_t> verts(static_cast<size_t>(n) * k);
        std::vector<double> vals(static_cast<size_t>(n));
        rd(verts.data(), verts.size() * 8);
        rd(vals.data(), vals.size() * 8);
        cells.reserve(cells.size() + n);
        for (std::int64_t i = 0; i < n; ++i) {
            typename Simp::IdxVector vs(verts.begin() + i * k, verts.begin() + (i + 1) * k);
            std::sort(vs.begin(), vs.end());
            cells.emplace_back(Simp(oineus::presorted_t{}, std::move(vs)), vals[i]);
        }
    }
    std::fclose(f);
    return Fil(std::move(cells), false, 8);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s (fr N | vr N | load FILE) MODE [threads_csv] [reps]\n", argv[0]);
        return 1;
    }
    const std::string kind = argv[1];
    const std::string arg = argv[2];
    const std::string mode = argv[3];
    if (mode != "col" && mode != "row" && mode != "default") {
        std::fprintf(stderr, "MODE must be col, row or default (bucket is timed alongside)\n");
        return 1;
    }
    // latch the forced scatter mode before the first transpose call
    // ("default" leaves the production routing in charge)
    if (mode != "default")
        setenv("OINEUS_TRANSPOSE_MODE", mode.c_str(), 1);

    std::vector<int> thread_counts{1, 2, 4, 8, 16};
    if (argc > 4) {
        thread_counts.clear();
        for (const char* p = argv[4]; *p;) {
            thread_counts.push_back(std::atoi(p));
            p = std::strchr(p, ',');
            if (!p) break;
            ++p;
        }
    }
    const int reps = argc > 5 ? std::atoi(argv[5]) : 3;

    std::optional<Fil> fil;
    std::string matrix_name;
    if (kind == "fr") {
        const Int n = std::atol(arg.c_str());
        matrix_name = "freudenthal_" + arg + "^3";
        std::vector<Real> data(static_cast<size_t>(n) * n * n);
        std::mt19937_64 gen(1);
        std::uniform_real_distribution<Real> dist(0.0, 1.0);
        for (auto& x : data) x = dist(gen);
        using Grid = oineus::Grid<Int, Real, 3>;
        typename Grid::GridPoint dims{n, n, n};
        Grid grid(dims, false, data.data(), Grid::DataLocation::VERTEX);
        fil = grid.freudenthal_filtration(3, false, 8);
    } else if (kind == "vr") {
        const size_t n = std::atol(arg.c_str());
        matrix_name = "vr_" + arg + "pts";
        std::vector<oineus::Point<Real, 3>> points(n);
        std::mt19937_64 gen(2);
        std::uniform_real_distribution<Real> dist(0.0, 1.0);
        for (auto& p : points)
            for (auto& x : p) x = dist(gen);
        fil = oineus::get_vr_filtration<Int, Real, 3>(points, 2,
                std::numeric_limits<Real>::max(), 8);
    } else if (kind == "load") {
        matrix_name = "loaded";
        fil = load_filtration(arg.c_str());
    } else {
        std::fprintf(stderr, "unknown input kind %s\n", kind.c_str());
        return 1;
    }

    std::fprintf(stderr, "# %s: %ld cells, max_dim %d, mode %s\n",
            matrix_name.c_str(), static_cast<long>(fil->size()), int(fil->max_dim()), mode.c_str());

    // production pattern 3: anti-transpose of the boundary matrix D (the
    // cohomology path when no direct coboundary is available, e.g. alpha).
    // Compared against a bucket-transpose emulation (independent of the
    // reduction, so benchmarked once per input).
    {
        auto bdry = fil->boundary_matrix(8);
        bench_antitranspose(matrix_name, bdry, bdry.size(), thread_counts, reps);
    }

    for (bool dualize : {false, true}) {
        const char* side = dualize ? "coh" : "hom";
        oineus::VRUDecomposition<Int> dcmp(*fil, dualize);
        oineus::ReductionParams params;
        params.compute_v = true;
        params.use_clearing = true;
        params.n_threads = 8;
        dcmp.reduce(params);
        // fill u_data_t via the production per-dim U solve (the in-reduce
        // compute_u path is serial-only)
        for (oineus::dim_type d = 0; d <= fil->max_dim(); ++d)
            dcmp.compute_u_from_v_1(d, 8, false);

        const Int n = static_cast<Int>(dcmp.v_data.size());

        // production pattern 1: full-matrix V transpose (ensure_ri_v_/make_dynamic)
        std::vector<std::pair<int, std::pair<size_t, size_t>>> full_job{
                {-1, {size_t(0), dcmp.v_data.size()}}};
        bench_matrix(matrix_name.c_str(), side, mode.c_str(), dcmp.v_data, n,
                full_job, thread_counts, reps);

        // production pattern 2: per-dimension U transpose (compute_u). Recover
        // the column form of U by transposing the at-rest row form once.
        MatrixData u_cols = ST::col_to_row_format_bucket(dcmp.u_data_t, 8, 0,
                dcmp.u_data_t.size(), n);
        u_cols.resize(dcmp.v_data.size());
        std::vector<std::pair<int, std::pair<size_t, size_t>>> dim_jobs;
        for (oineus::dim_type d = 0; d <= fil->max_dim(); ++d)
            dim_jobs.emplace_back(int(d),
                    std::make_pair(size_t(fil->dim_first(d)), size_t(fil->dim_last(d) + 1)));
        bench_matrix((matrix_name + "_U").c_str(), side, mode.c_str(), u_cols, n,
                dim_jobs, thread_counts, reps);

        // production pattern 4: per-dimension V transpose (compute_u_from_v,
        // the VTUT solve V^T U^T = Id, which needs V^T up front; V is
        // block-diagonal in dim, so production transposes one dim block)
        bench_matrix((matrix_name + "_Vt").c_str(), side, mode.c_str(), dcmp.v_data, n,
                dim_jobs, thread_counts, reps);
    }
    return 0;
}
