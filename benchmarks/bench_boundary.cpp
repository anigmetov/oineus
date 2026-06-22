// Benchmark: boundary / coboundary MATRIX CONSTRUCTION, current master vs a
// compact "packed-uid" simplex representation.
//
// Idea under test (per user request): a filtration does not store a vector of
// standalone simplices with an explicit std::vector<vertex>. Instead each cell
// is a single packed integer (the uid); the per-filtration-type domain info
// lives once in the filtration, and a policy computes boundary/coboundary
// directly from the packed uid -- no per-face vector allocation, and, for
// Freudenthal/cubical, a DIRECT coboundary that skips the antitranspose.
//
// Encodings benchmarked:
//   VR          : (a) Bauer combinatorial uid (unrank-on-demand, like Ripser)
//                 (b) simple bit packing: k bits per vertex id
//                 master = current Simplex<Int> (explicit sorted vertex vector)
//   Freudenthal : packed (anchor vertex id << type_bits | simplex-type); a
//                 precomputed table drives boundary AND coboundary directly.
//                 master = current Simplex<Int> (vector of vertices).
//   Cubical     : packed cube uid (anchor<<3 | face-bits) + shared domain;
//                 direct coboundary. master = current Cube<Int,D> (per-cell
//                 domain copy) + antitranspose coboundary.
//
// For Freudenthal/cubical the geometric uid is a small dense integer, so the
// uid->sorted_id map can be a flat direct-address array instead of a hash map;
// both are timed (hash vs flat) as an ablation.
//
// All timings are single-threaded to isolate per-column cost (construction is
// embarrassingly parallel; the direct coboundary is additionally lock-free per
// column, unlike the antitranspose). Every "new" matrix is verified against the
// master ground truth before its timing is reported.
//
//   ./bench_boundary                       # vr + grid(freudenthal) + cube
//   ./bench_boundary --only vr --n-points 250 --vr-max-dim 3
//   ./bench_boundary --only grid --grid-side 72
//   ./bench_boundary --only cube --grid-side 110 --reps 7

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <oineus/oineus.h>

using Int = int;
using Real = double;
using Uid128 = unsigned __int128;
using Col = oineus::SparseColumn<Int>;
using MatrixData = std::vector<Col>;

// ----------------------------------------------------------------------------
// small utilities
// ----------------------------------------------------------------------------

struct Hash128 {
    std::size_t operator()(Uid128 v) const noexcept
    {
        auto lo = static_cast<std::uint64_t>(v);
        auto hi = static_cast<std::uint64_t>(v >> 64);
        std::size_t s = std::hash<std::uint64_t>{}(lo);
        s ^= std::hash<std::uint64_t>{}(hi) + 0x9e3779b97f4a7c15ULL + (s << 6) + (s >> 2);
        return s;
    }
};

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// Time a callable that rebuilds a matrix into `out` each call. One warm-up,
// then median of `reps`. `out` is left holding the final result.
template<class F>
double time_ms(int reps, F&& f)
{
    f();
    std::vector<double> ts;
    for (int r = 0; r < reps; ++r) {
        Timer tm;
        f();
        ts.push_back(tm.elapsed() * 1000.0);
    }
    return median(ts);
}

static size_t nnz(const MatrixData& m)
{
    size_t s = 0;
    for (const auto& c : m) s += c.size();
    return s;
}

static bool equal_mat(const MatrixData& a, const MatrixData& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j)
            if (a[i][j] != b[i][j]) return false;
    }
    return true;
}

// Ground-truth coboundary relation = plain transpose of the boundary matrix:
// cob[r] = sorted { c : r in bd[c] }. (NOT the antitranspose, whose indices are
// reversed; this is the unambiguous reference for the direct coboundary.)
static MatrixData transpose_relation(const MatrixData& bd, size_t n)
{
    MatrixData cob(n);
    for (size_t c = 0; c < bd.size(); ++c)
        for (Int r : bd[c])
            cob[r].push_back(static_cast<Int>(c));
    for (auto& col : cob)
        std::sort(col.begin(), col.end());
    return cob;
}

// is any entry negative (a failed map lookup leaking -1)?
static bool has_negative(const MatrixData& m)
{
    for (const auto& c : m)
        for (Int x : c)
            if (x < 0) return true;
    return false;
}

// ============================================================================
// VR
// ============================================================================

static void bench_vr(size_t n_points, int max_dim, int reps)
{
    std::cout << "\n================ VR (Vietoris-Rips) ================\n";
    std::cout << "n_points=" << n_points << " max_dim=" << max_dim << "\n";

    std::mt19937_64 gen(42);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::vector<oineus::Point<Real, 3>> points(n_points);
    for (auto& p : points)
        for (int j = 0; j < 3; ++j) p[j] = dist(gen);

    auto fil = oineus::get_vr_filtration<Int, Real, 3>(points, static_cast<size_t>(max_dim),
            std::numeric_limits<Real>::max(), /*n_threads=*/8);
    const size_t n = fil.size();
    std::cout << "filtration size (n_cols): " << n << "\n";
    using VRCell = std::decay_t<decltype(fil.cells()[0])>;
    std::cout << "per-cell footprint: master sizeof(cell)=" << sizeof(VRCell)
              << " B + heap (dim+1)*" << sizeof(Int) << " B vertex array;"
              << " packed = 16 B uid (128-bit), no heap\n";

    // ---- packed representations (built untimed, like master's prebuilt map) ----
    const int DIMSHIFT = 124;
    const Uid128 FIELD_MASK = (Uid128(1) << DIMSHIFT) - 1;

    // bits per vertex for the bit-packing scheme
    int k = 1;
    while ((Uid128(1) << k) < static_cast<Uid128>(n_points)) ++k;
    if (static_cast<long>(max_dim + 1) * k > DIMSHIFT) {
        std::cout << "bit-packing does not fit in 124 bits; skipping VR\n";
        return;
    }

    // binomials C(v,kk) matching oineus::simplex_uid (comb), for unranking Bauer
    std::vector<std::vector<Uid128>> colC(max_dim + 2, std::vector<Uid128>(n_points));
    for (int kk = 0; kk <= max_dim + 1; ++kk)
        for (size_t v = 0; v < n_points; ++v)
            colC[kk][v] = oineus::comb<Int, Uid128>(static_cast<Int>(v), kk);

    std::vector<Uid128> bit_uid(n), bau_uid(n);
    std::vector<int> dims(n);
    std::unordered_map<Uid128, Int, Hash128> bit_map;
    bit_map.reserve(n * 2);

    for (size_t c = 0; c < n; ++c) {
        const auto& cell = fil.cells()[c];
        const auto& vs = cell.get_cell().get_vertices();
        int d = cell.dim();
        dims[c] = d;
        int nverts = d + 1;
        Uid128 f = 0;
        for (int j = 0; j < nverts; ++j)
            f |= static_cast<Uid128>(static_cast<std::uint64_t>(vs[j])) << (j * k);
        Uid128 bu = f | (static_cast<Uid128>(nverts + 1) << DIMSHIFT);
        bit_uid[c] = bu;
        bau_uid[c] = cell.get_uid();   // == oineus::simplex_uid(sorted vs)
        bit_map[bu] = static_cast<Int>(c);
    }

    // ---- timed builds ----
    MatrixData master_bd, bit_bd, bau_bd;

    double t_master = time_ms(reps, [&]() { master_bd = fil.boundary_matrix(1); });

    double t_bit = time_ms(reps, [&]() {
        MatrixData res(n);
        for (size_t c = 0; c < n; ++c) {
            int d = dims[c];
            if (d > 0) {
                Uid128 fields = bit_uid[c] & FIELD_MASK;
                int nverts = d + 1;
                Col col;
                for (int i = 0; i < nverts; ++i) {
                    Uid128 lowmask = (Uid128(1) << (i * k)) - 1;
                    Uid128 low = fields & lowmask;
                    Uid128 high = fields >> ((i + 1) * k);
                    Uid128 facet = (low | (high << (i * k))) | (static_cast<Uid128>(nverts) << DIMSHIFT);
                    col.push_back(bit_map.at(facet));
                }
                std::sort(col.begin(), col.end());
                res[c] = std::move(col);
            }
        }
        bit_bd.swap(res);
    });

    double t_bau = time_ms(reps, [&]() {
        MatrixData res(n);
        std::vector<Int> verts(max_dim + 1);
        for (size_t c = 0; c < n; ++c) {
            int d = dims[c];
            if (d > 0) {
                Uid128 idx = bau_uid[c] & FIELD_MASK;
                for (int pos = d; pos >= 0; --pos) {
                    const auto& cc = colC[pos + 1];
                    Int v = static_cast<Int>(std::upper_bound(cc.begin(), cc.end(), idx) - cc.begin()) - 1;
                    verts[pos] = v;
                    idx -= cc[v];
                }
                Col col;
                for (int i = 0; i <= d; ++i) {
                    Uid128 ff = 0;
                    for (int j = 0; j <= d; ++j) {
                        if (j < i) ff += colC[j + 1][verts[j]];
                        else if (j > i) ff += colC[j][verts[j]];
                    }
                    Uid128 facet = ff | (static_cast<Uid128>(d + 1) << DIMSHIFT);
                    col.push_back(fil.uid_to_sorted_id.at(facet));
                }
                std::sort(col.begin(), col.end());
                res[c] = std::move(col);
            }
        }
        bau_bd.swap(res);
    });

    // master coboundary (antitranspose) -- VR has no direct coboundary
    MatrixData master_cob;
    double t_cob = time_ms(reps, [&]() { master_cob = fil.coboundary_matrix(1); });

    bool ok_bit = equal_mat(bit_bd, master_bd) and not has_negative(bit_bd);
    bool ok_bau = equal_mat(bau_bd, master_bd) and not has_negative(bau_bd);

    std::cout << "boundary nnz: " << nnz(master_bd) << "\n";
    std::cout << "\nBOUNDARY build (ms, single-thread, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (vector<vertex> Simplex)", t_master);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new: bit-packed uid", t_bit,
            t_bit > 0 ? t_master / t_bit : 0.0, ok_bit ? "OK" : "MISMATCH");
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new: Bauer uid (unrank)", t_bau,
            t_bau > 0 ? t_master / t_bau : 0.0, ok_bau ? "OK" : "MISMATCH");
    std::cout << "\nCOBOUNDARY build (ms): master antitranspose = " << t_cob
              << "  (VR: no direct coboundary -- needs the neighbor graph; antitranspose kept)\n";
}

// ============================================================================
// Freudenthal  (D = 3)
// ============================================================================

template<size_t D>
using Pt = std::array<Int, D>;

template<size_t D>
static Pt<D> min_corner(const std::vector<Pt<D>>& s)
{
    Pt<D> m = s[0];
    for (const auto& p : s)
        for (size_t d = 0; d < D; ++d)
            m[d] = std::min(m[d], p[d]);
    return m;
}

template<size_t D>
static std::vector<Pt<D>> normalize(std::vector<Pt<D>> s)
{
    Pt<D> m = min_corner<D>(s);
    for (auto& p : s)
        for (size_t d = 0; d < D; ++d) p[d] -= m[d];
    std::sort(s.begin(), s.end());
    return s;
}

template<size_t D>
static void bench_freudenthal(size_t side, int reps)
{
    std::cout << "\n============ Freudenthal grid (D=" << D << ") ============\n";
    std::cout << "grid side=" << side << " top_d=" << D << "\n";

    using Grid = oineus::Grid<Int, Real, D>;
    typename Grid::GridPoint dims;
    size_t total = 1;
    for (size_t d = 0; d < D; ++d) { dims[d] = static_cast<Int>(side); total *= side; }

    std::mt19937_64 gen(7);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::vector<Real> data(total);
    for (auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.freudenthal_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/8);
    const auto& dom = grid.domain();
    const size_t n = fil.size();
    std::cout << "filtration size (n_cols): " << n << "\n";
    using FrCell = std::decay_t<decltype(fil.cells()[0])>;
    std::cout << "per-cell footprint: master sizeof(cell)=" << sizeof(FrCell)
              << " B + heap (dim+1)*" << sizeof(Int) << " B vertex array;"
              << " packed = 8 B (anchor<<type_bits | type), no heap\n";

    // ---- build the simplex-type tables from fr_displacements ----
    std::map<std::vector<Pt<D>>, int> set2type;
    std::vector<int> type_dim;
    std::vector<std::vector<Pt<D>>> type_disps;
    std::vector<Pt<D>> type_maxoff;

    auto register_type = [&](const std::vector<Pt<D>>& norm, int d) {
        auto it = set2type.find(norm);
        if (it != set2type.end()) return it->second;
        int id = static_cast<int>(type_dim.size());
        set2type[norm] = id;
        type_dim.push_back(d);
        type_disps.push_back(norm);
        Pt<D> mo{};
        for (size_t dd = 0; dd < D; ++dd) {
            Int mx = 0;
            for (const auto& p : norm) mx = std::max(mx, p[dd]);
            mo[dd] = mx;
        }
        type_maxoff.push_back(mo);
        return id;
    };

    for (int d = 0; d <= static_cast<int>(D); ++d) {
        auto disps = dom.get_fr_displacements(d);   // vector<vector<GridPoint>>
        for (const auto& pattern : disps) {
            std::vector<Pt<D>> s(pattern.begin(), pattern.end());
            register_type(normalize<D>(s), d);
        }
    }
    int n_types = static_cast<int>(type_dim.size());
    int type_bits = 1;
    while ((1 << type_bits) < n_types) ++type_bits;
    Int type_mask = (Int(1) << type_bits) - 1;

    // boundary table: per type, list of (id_offset, point-offset m, facet_type)
    struct BdEntry { Int id_off; Pt<D> m; int ft; };
    std::vector<std::vector<BdEntry>> bd_table(n_types);
    for (int t = 0; t < n_types; ++t) {
        int d = type_dim[t];
        if (d == 0) continue;
        for (int i = 0; i <= d; ++i) {
            std::vector<Pt<D>> rem;
            for (int j = 0; j <= d; ++j) if (j != i) rem.push_back(type_disps[t][j]);
            Pt<D> m = min_corner<D>(rem);
            auto norm = normalize<D>(rem);
            int ft = set2type.at(norm);
            bd_table[t].push_back({dom.point_to_id(m), m, ft});
        }
    }

    // coboundary table: invert the boundary table.
    struct CobEntry { Pt<D> delta; Pt<D> maxoff; int cob_type; };
    std::vector<std::vector<CobEntry>> cob_table(n_types);
    for (int T = 0; T < n_types; ++T)
        for (const auto& e : bd_table[T])
            cob_table[e.ft].push_back({e.m, type_maxoff[T], T});

    // ---- pack every cell + build maps (untimed) ----
    std::vector<std::int64_t> packed(n);
    std::vector<int> cdim(n);
    std::unordered_map<std::int64_t, Int> hash_map;
    hash_map.reserve(n * 2);
    std::vector<Int> flat(static_cast<size_t>(total) << type_bits, -1);

    for (size_t c = 0; c < n; ++c) {
        const auto& cell = fil.cells()[c];
        const auto& vs = cell.get_cell().get_vertices();
        int d = cell.dim();
        cdim[c] = d;
        Int anchor_id = vs[0];
        Pt<D> ap = dom.id_to_point(anchor_id);
        std::vector<Pt<D>> s;
        s.reserve(vs.size());
        for (Int v : vs) {
            Pt<D> p = dom.id_to_point(v);
            for (size_t dd = 0; dd < D; ++dd) p[dd] -= ap[dd];
            s.push_back(p);
        }
        int t = set2type.at(normalize<D>(s));
        std::int64_t pk = (static_cast<std::int64_t>(anchor_id) << type_bits) | t;
        packed[c] = pk;
        cdim[c] = d;
        hash_map[pk] = static_cast<Int>(c);
        flat[pk] = static_cast<Int>(c);
    }

    Pt<D> shape = dom.shape();

    // ---- boundary: master, new(hash), new(flat) ----
    MatrixData master_bd, new_bd_hash, new_bd_flat;
    double t_master_bd = time_ms(reps, [&]() { master_bd = fil.boundary_matrix(1); });

    auto build_bd = [&](MatrixData& out, auto&& lookup) {
        MatrixData res(n);
        for (size_t c = 0; c < n; ++c) {
            int d = cdim[c];
            if (d > 0) {
                Int anchor_id = static_cast<Int>(packed[c] >> type_bits);
                int t = static_cast<int>(packed[c] & type_mask);
                Col col;
                for (const auto& e : bd_table[t]) {
                    std::int64_t facet = (static_cast<std::int64_t>(anchor_id + e.id_off) << type_bits) | e.ft;
                    col.push_back(lookup(facet));
                }
                std::sort(col.begin(), col.end());
                res[c] = std::move(col);
            }
        }
        out.swap(res);
    };

    double t_bd_hash = time_ms(reps, [&]() { build_bd(new_bd_hash, [&](std::int64_t u) { return hash_map.at(u); }); });
    double t_bd_flat = time_ms(reps, [&]() { build_bd(new_bd_flat, [&](std::int64_t u) { return flat[u]; }); });

    // ---- coboundary: master(antitranspose), new direct(hash), new direct(flat) ----
    MatrixData master_cob, new_cob_hash, new_cob_flat;
    double t_master_cob = time_ms(reps, [&]() { master_cob = fil.coboundary_matrix(1); });

    auto build_cob = [&](MatrixData& out, auto&& lookup) {
        MatrixData res(n);
        for (size_t c = 0; c < n; ++c) {
            Int anchor_id = static_cast<Int>(packed[c] >> type_bits);
            int t = static_cast<int>(packed[c] & type_mask);
            if (not cob_table[t].empty()) {
                Pt<D> ap = dom.id_to_point(anchor_id);
                Col col;
                for (const auto& e : cob_table[t]) {
                    Pt<D> ca;
                    bool ok = true;
                    for (size_t dd = 0; dd < D; ++dd) {
                        ca[dd] = ap[dd] - e.delta[dd];
                        if (ca[dd] < 0 or ca[dd] + e.maxoff[dd] > shape[dd] - 1) { ok = false; break; }
                    }
                    if (ok) {
                        Int cid = dom.point_to_id(ca);
                        std::int64_t cof = (static_cast<std::int64_t>(cid) << type_bits) | e.cob_type;
                        col.push_back(lookup(cof));
                    }
                }
                std::sort(col.begin(), col.end());
                res[c] = std::move(col);
            }
        }
        out.swap(res);
    };

    double t_cob_hash = time_ms(reps, [&]() { build_cob(new_cob_hash, [&](std::int64_t u) { return hash_map.at(u); }); });
    double t_cob_flat = time_ms(reps, [&]() { build_cob(new_cob_flat, [&](std::int64_t u) { return flat[u]; }); });

    MatrixData cob_truth = transpose_relation(master_bd, n);

    bool ok_bd_hash = equal_mat(new_bd_hash, master_bd) and not has_negative(new_bd_hash);
    bool ok_bd_flat = equal_mat(new_bd_flat, master_bd) and not has_negative(new_bd_flat);
    bool ok_cob_hash = equal_mat(new_cob_hash, cob_truth) and not has_negative(new_cob_hash);
    bool ok_cob_flat = equal_mat(new_cob_flat, cob_truth) and not has_negative(new_cob_flat);

    std::cout << "n_types=" << n_types << " type_bits=" << type_bits
              << "  boundary nnz=" << nnz(master_bd) << "  coboundary nnz=" << nnz(cob_truth) << "\n";

    std::cout << "\nBOUNDARY build (ms, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (vector<vertex> Simplex)", t_master_bd);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new packed, hash map", t_bd_hash,
            t_bd_hash > 0 ? t_master_bd / t_bd_hash : 0.0, ok_bd_hash ? "OK" : "MISMATCH");
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new packed, flat map", t_bd_flat,
            t_bd_flat > 0 ? t_master_bd / t_bd_flat : 0.0, ok_bd_flat ? "OK" : "MISMATCH");

    std::cout << "\nCOBOUNDARY build (ms, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (antitranspose)", t_master_cob);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new DIRECT, hash map", t_cob_hash,
            t_cob_hash > 0 ? t_master_cob / t_cob_hash : 0.0, ok_cob_hash ? "OK" : "MISMATCH");
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new DIRECT, flat map", t_cob_flat,
            t_cob_flat > 0 ? t_master_cob / t_cob_flat : 0.0, ok_cob_flat ? "OK" : "MISMATCH");
}

// ============================================================================
// Cubical  (D = 3)
// ============================================================================

template<size_t D, class MapFn>
static Col cube_boundary_col(Int uid, const oineus::GridDomain<Int, D>& dom, MapFn&& M)
{
    constexpr int MC = OINEUS_MAX_CUBE_DIM;
    Col col;
    int bits[MC];
    int nb = 0;
    for (int d = 0; d < MC; ++d)
        if (uid & (1 << d)) bits[nb++] = d;
    if (nb == 0) return col;
    for (int kk = 0; kk < nb; ++kk)
        col.push_back(M(uid & ~(Int(1) << bits[kk])));
    Int vpart = uid >> MC;
    auto anchor = dom.id_to_point(vpart);
    for (int kk = 0; kk < nb; ++kk) {
        auto p = anchor;
        p[bits[kk]] += 1;
        Int face_id = dom.point_to_id(p) << MC;
        for (int j = 0; j < nb; ++j)
            if (j != kk) face_id |= (Int(1) << bits[j]);
        col.push_back(M(face_id));
    }
    std::sort(col.begin(), col.end());
    return col;
}

template<size_t D, class MapFn>
static Col cube_coboundary_col(Int uid, const oineus::GridDomain<Int, D>& dom, MapFn&& M)
{
    constexpr int MC = OINEUS_MAX_CUBE_DIM;
    Col col;
    Int vertex_id = uid >> MC;
    auto vertex = dom.id_to_point(vertex_id);
    Int cube_bits = uid & ((1 << MC) - 1);
    for (unsigned d = 0; d < D; ++d) {
        if (!(cube_bits & (1 << d))) {
            auto opp = vertex;
            for (unsigned dd = 0; dd < D; ++dd)
                if ((cube_bits & (1 << dd)) || dd == d) opp[dd] += 1;
            if (dom.contains(opp))
                col.push_back(M((vertex_id << MC) | cube_bits | (1 << d)));
        }
    }
    for (unsigned d = 0; d < D; ++d) {
        if (!(cube_bits & (1 << d))) {
            auto sv = vertex;
            sv[d] -= 1;
            if (dom.contains(sv)) {
                auto opp = sv;
                for (unsigned dd = 0; dd < D; ++dd)
                    if ((cube_bits & (1 << dd)) || dd == d) opp[dd] += 1;
                if (dom.contains(opp)) {
                    Int svid = dom.point_to_id(sv);
                    col.push_back(M((svid << MC) | cube_bits | (1 << d)));
                }
            }
        }
    }
    std::sort(col.begin(), col.end());
    return col;
}

template<size_t D>
static void bench_cubical(size_t side, int reps)
{
    std::cout << "\n============ Cubical grid (D=" << D << ") ============\n";
    std::cout << "grid side=" << side << " top_d=" << D << "\n";

    using Grid = oineus::Grid<Int, Real, D>;
    typename Grid::GridPoint dims;
    size_t total = 1;
    for (size_t d = 0; d < D; ++d) { dims[d] = static_cast<Int>(side); total *= side; }

    std::mt19937_64 gen(11);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::vector<Real> data(total);
    for (auto& x : data) x = dist(gen);

    Grid grid(dims, /*wrap=*/false, data.data(), Grid::DataLocation::VERTEX);
    auto fil = grid.cube_filtration(/*top_d=*/D, /*negate=*/false, /*n_threads=*/8);
    const auto& dom = grid.domain();
    const size_t n = fil.size();
    std::cout << "filtration size (n_cols): " << n << "\n";
    using CubeCell = std::decay_t<decltype(fil.cells()[0])>;
    std::cout << "per-cell footprint: master sizeof(cell)=" << sizeof(CubeCell)
              << " B (includes a per-cell GridDomain copy); packed = "
              << sizeof(Int) << " B uid + shared domain, no per-cell domain\n";

    constexpr int MC = OINEUS_MAX_CUBE_DIM;
    std::vector<Int> packed(n);
    std::unordered_map<Int, Int> hash_map;
    hash_map.reserve(n * 2);
    std::vector<Int> flat(static_cast<size_t>(dom.size()) << MC, -1);

    for (size_t c = 0; c < n; ++c) {
        Int uid = fil.cells()[c].get_cell().get_uid();
        packed[c] = uid;
        hash_map[uid] = static_cast<Int>(c);
        flat[uid] = static_cast<Int>(c);
    }

    // boundary
    MatrixData master_bd, new_bd_hash, new_bd_flat;
    double t_master_bd = time_ms(reps, [&]() { master_bd = fil.boundary_matrix(1); });

    auto build_bd = [&](MatrixData& out, auto&& lookup) {
        MatrixData res(n);
        for (size_t c = 0; c < n; ++c)
            res[c] = cube_boundary_col<D>(packed[c], dom, lookup);
        out.swap(res);
    };
    double t_bd_hash = time_ms(reps, [&]() { build_bd(new_bd_hash, [&](Int u) { return hash_map.at(u); }); });
    double t_bd_flat = time_ms(reps, [&]() { build_bd(new_bd_flat, [&](Int u) { return flat[u]; }); });

    // coboundary
    MatrixData master_cob, new_cob_hash, new_cob_flat;
    double t_master_cob = time_ms(reps, [&]() { master_cob = fil.coboundary_matrix(1); });

    auto build_cob = [&](MatrixData& out, auto&& lookup) {
        MatrixData res(n);
        for (size_t c = 0; c < n; ++c)
            res[c] = cube_coboundary_col<D>(packed[c], dom, lookup);
        out.swap(res);
    };
    double t_cob_hash = time_ms(reps, [&]() { build_cob(new_cob_hash, [&](Int u) { return hash_map.at(u); }); });
    double t_cob_flat = time_ms(reps, [&]() { build_cob(new_cob_flat, [&](Int u) { return flat[u]; }); });

    MatrixData cob_truth = transpose_relation(master_bd, n);

    bool ok_bd_hash = equal_mat(new_bd_hash, master_bd) and not has_negative(new_bd_hash);
    bool ok_bd_flat = equal_mat(new_bd_flat, master_bd) and not has_negative(new_bd_flat);
    bool ok_cob_hash = equal_mat(new_cob_hash, cob_truth) and not has_negative(new_cob_hash);
    bool ok_cob_flat = equal_mat(new_cob_flat, cob_truth) and not has_negative(new_cob_flat);

    std::cout << "boundary nnz=" << nnz(master_bd) << "  coboundary nnz=" << nnz(cob_truth) << "\n";

    std::cout << "\nBOUNDARY build (ms, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (Cube w/ per-cell domain)", t_master_bd);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new packed uid, hash map", t_bd_hash,
            t_bd_hash > 0 ? t_master_bd / t_bd_hash : 0.0, ok_bd_hash ? "OK" : "MISMATCH");
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new packed uid, flat map", t_bd_flat,
            t_bd_flat > 0 ? t_master_bd / t_bd_flat : 0.0, ok_bd_flat ? "OK" : "MISMATCH");

    std::cout << "\nCOBOUNDARY build (ms, median of " << reps << "):\n";
    std::printf("  %-34s %9.2f\n", "master (antitranspose)", t_master_cob);
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new DIRECT, hash map", t_cob_hash,
            t_cob_hash > 0 ? t_master_cob / t_cob_hash : 0.0, ok_cob_hash ? "OK" : "MISMATCH");
    std::printf("  %-34s %9.2f   speedup %5.2fx   %s\n", "new DIRECT, flat map", t_cob_flat,
            t_cob_flat > 0 ? t_master_cob / t_cob_flat : 0.0, ok_cob_flat ? "OK" : "MISMATCH");
}

// ============================================================================

int main(int argc, char** argv)
{
    std::string only = "all";
    size_t n_points = 220;
    int vr_max_dim = 3;
    size_t grid_side = 64;
    int reps = 5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if (a == "--only") only = need();
        else if (a == "--n-points") n_points = std::stoul(need());
        else if (a == "--vr-max-dim") vr_max_dim = std::stoi(need());
        else if (a == "--grid-side") grid_side = std::stoul(need());
        else if (a == "--reps") reps = std::stoi(need());
        else { std::cerr << "unknown arg: " << a << "\n"; return 1; }
    }

    if (only == "all" or only == "vr")
        bench_vr(n_points, vr_max_dim, reps);
    if (only == "all" or only == "grid" or only == "freudenthal")
        bench_freudenthal<3>(grid_side, reps);
    if (only == "all" or only == "cube" or only == "cubical")
        bench_cubical<3>(grid_side, reps);

    return 0;
}
