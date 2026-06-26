// Benchmark for parallel persistence-diagram extraction.
//
// Builds a large filtration (random-valued 3D Freudenthal grid, or VR of random
// points), reduces it once (fused by default; the reduction is NOT timed), then
// times three diagram extractions -- serial, taskflow-parallel, raw-std::thread
// parallel -- at several thread counts, reporting parallel speedup and the
// taskflow-vs-std::thread scheduler overhead. All three are checked to produce
// identical diagrams before timing.
//
// Examples:
//   ./bench_diagram                              # grid, side 90 (~10M cells)
//   ./bench_diagram --grid-side 50 --reps 7
//   ./bench_diagram --mode vr --n-points 400
//   ./bench_diagram --threads 1,2,4,8,16 --classic --dualize
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <oineus/oineus.h>

using Int = int;
using Real = double;

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

template<class Fil>
int run_bench(const Fil& fil, bool fused, bool dualize, const std::vector<int>& threads, int reps, int reduce_threads)
{
    const size_t n_cols = fil.size();
    std::cout << "n_cols (filtration size): " << n_cols << "\n";

    oineus::Params params;
    params.n_threads = reduce_threads;
    params.compute_v = false;

    auto make_dcmp = [&]() {
        if (fused)
            return oineus::VRUDecomposition<Int>::reduce_from_filtration_fused(fil, params, dualize);
        oineus::VRUDecomposition<Int> d(fil, dualize);
        d.reduce(params);
        return d;
    };
    Timer t_reduce;
    auto dcmp = make_dcmp();
    std::cout << "reduce (" << (fused ? "fused" : "classic") << ", " << reduce_threads
              << " threads): " << t_reduce.elapsed() * 1000 << " ms [excluded from timing]\n";
    std::cout << "state: " << (dcmp.is_pivots_only() ? "fused / pivots-only" : "classic / r_data") << "\n";

    // The three diagram-extraction variants now take the cell-type-erased FiltrationValues
    // view; build it once here so the timing measures the extraction, not the (common) view
    // build (which is the same fixed cost the public diagram() pays per call).
    auto fv = fil.values_view();

    // include_all = false, include_inf_points = true, only_zero_persistence = false
    auto serial_dgm = [&]() { return dcmp.diagram_general_serial(fv, false, true, false); };
    auto tf_dgm     = [&](int nt) { return dcmp.diagram_general_par(fv, false, true, false, nt); };
    auto st_dgm     = [&](int nt) { return dcmp.diagram_general_par_stdthread(fv, false, true, false, nt); };

    // warm up (page in cells_, etc.)
    { auto w = serial_dgm(); (void) w; }

    // parity: all three must agree as a sorted multiset before we trust timings.
    {
        auto ds = serial_dgm();
        auto dp = tf_dgm(threads.back());
        auto dt = st_dgm(threads.back());
        ds.sort(); dp.sort(); dt.sort();
        bool ok = (ds == dp) and (ds == dt);
        std::cout << "parity (serial == taskflow == stdthread): " << (ok ? "OK" : "FAIL") << "\n";
        size_t total = 0;
        for (size_t d = 0; d < ds.n_dims(); ++d)
            total += ds[d].size();
        std::cout << "diagram points (off-diagonal + essential): " << total << "\n";
        if (not ok)
            return 2;
    }

    std::vector<double> st;
    for (int r = 0; r < reps; ++r) {
        Timer t;
        auto d = serial_dgm();
        st.push_back(t.elapsed());
        (void) d;
    }
    const double serial_ms = median(st) * 1000;
    std::cout << "\nserial baseline: " << serial_ms << " ms (median of " << reps << ")\n\n";

    std::printf("%-8s %13s %14s %12s %12s %14s\n",
            "threads", "taskflow_ms", "stdthread_ms", "speedup_tf", "speedup_st", "tf_overhead_%");
    for (int nt : threads) {
        std::vector<double> tf, th;
        for (int r = 0; r < reps; ++r) { Timer t; auto d = tf_dgm(nt); tf.push_back(t.elapsed()); (void) d; }
        for (int r = 0; r < reps; ++r) { Timer t; auto d = st_dgm(nt); th.push_back(t.elapsed()); (void) d; }
        const double tf_ms = median(tf) * 1000;
        const double th_ms = median(th) * 1000;
        std::printf("%-8d %13.3f %14.3f %12.2f %12.2f %14.1f\n",
                nt, tf_ms, th_ms,
                tf_ms > 0 ? serial_ms / tf_ms : 0.0,
                th_ms > 0 ? serial_ms / th_ms : 0.0,
                th_ms > 0 ? 100.0 * (tf_ms - th_ms) / th_ms : 0.0);
    }
    return 0;
}

int main(int argc, char** argv)
{
    std::string mode = "grid";
    size_t grid_side = 90;   // 3D Freudenthal -> roughly 10M cells
    size_t n_points = 400;   // VR -> roughly 10M simplices at max_dim 2
    int vr_max_dim = 2;
    std::vector<int> threads = {1, 2, 4, 8};
    int reps = 5;
    int reduce_threads = 8;
    bool fused = true;
    bool dualize = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if (a == "--mode") mode = need();
        else if (a == "--grid-side") grid_side = std::stoul(need());
        else if (a == "--n-points") n_points = std::stoul(need());
        else if (a == "--vr-max-dim") vr_max_dim = std::stoi(need());
        else if (a == "--reps") reps = std::stoi(need());
        else if (a == "--reduce-threads") reduce_threads = std::stoi(need());
        else if (a == "--classic") fused = false;
        else if (a == "--dualize") dualize = true;
        else if (a == "--threads") {
            threads.clear();
            std::string s = need();
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (not tok.empty()) threads.push_back(std::stoi(tok));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        else { std::cerr << "unknown arg: " << a << "\n"; return 1; }
    }

    std::mt19937_64 gen(42);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    std::cout << "mode=" << mode << " dualize=" << dualize << " fused=" << fused << "\n";

    if (mode == "grid") {
        oineus::Grid<Int, Real, 3>::GridPoint dims{static_cast<Int>(grid_side), static_cast<Int>(grid_side), static_cast<Int>(grid_side)};
        std::vector<Real> data(grid_side * grid_side * grid_side);
        for (auto& x : data) x = dist(gen);
        oineus::Grid<Int, Real, 3> grid(dims, /*wrap=*/false, data.data(), oineus::Grid<Int, Real, 3>::DataLocation::VERTEX);
        auto fil = grid.freudenthal_filtration(/*top_d=*/3, /*negate=*/false, reduce_threads);
        return run_bench(fil, fused, dualize, threads, reps, reduce_threads);
    } else if (mode == "vr") {
        std::vector<oineus::Point<Real, 3>> points(n_points);
        for (auto& p : points)
            for (int j = 0; j < 3; ++j) p[j] = dist(gen);
        auto fil = oineus::get_vr_filtration<Int, Real, 3>(points, static_cast<size_t>(vr_max_dim),
                std::numeric_limits<Real>::max(), reduce_threads);
        return run_bench(fil, fused, dualize, threads, reps, reduce_threads);
    } else {
        std::cerr << "unknown mode: " << mode << " (use grid or vr)\n";
        return 1;
    }
}
