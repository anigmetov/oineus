#ifndef OINEUS_VIETORIS_RIPS_INORDER_H
#define OINEUS_VIETORIS_RIPS_INORDER_H

#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <cassert>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <thread>

#include "filtration.h"
#include "vietoris_rips.h"
#include "packed_simplex.h"
#include "interrupt.h"

// In-order Vietoris-Rips construction (VRE), following
//   Vejdemo-Johansson, Matuszewski, Bauer, "In-order generation of
//   Vietoris-Rips Complexes", arXiv:2411.05495.
//
// Each d-simplex is generated as the cofacet of a unique parent --
// its lexicographically first full-length facet. The parent carries
// a cached "longest edge" [s, t] (lex-first full-length edge of the
// parent, with s > t in the paper's descending convention) which is
// propagated unchanged to every cofacet (Caching Longest Edges
// Proposition in the paper).
//
// Throughout this file we use oineus's ascending vertex convention:
// vertices_ = [v_0, v_1, ..., v_{d-1}] with v_0 < ... < v_{d-1}.
// The paper writes simplices descending. Concretely, oineus's
// vertices_.back() is the paper's v_{d-1}.

namespace oineus {

namespace detail {

// Per-simplex working state during VRE layer-walk.
//
// VreFrame carries two distance-like values:
//
//   compare_value: a monotonic-in-distance quantity that the cofacet
//                  generation uses for its `<= alpha` and `< alpha`
//                  cutoff checks. The points-cloud path uses *squared*
//                  Euclidean distance here (cheap to compute, avoids
//                  per-pair sqrt). The dist-matrix path uses the *raw*
//                  user-supplied distance, so the cutoff comparison
//                  is exact and never overflows on inputs near
//                  std::numeric_limits<Real>::max().
//
//   diameter:      the value to store as the simplex's filtration
//                  value. For points it's sqrt(compare_value); for the
//                  dist-matrix path it equals the user's distance
//                  bit-for-bit.
//
// The two are propagated unchanged from parent to cofacet (cofacets
// inherit the parent's diameter exactly per VRE; see the Caching
// Longest Edges proposition).
//
// cached_edge stores the lex-first full-length edge (x = s, y = t,
// s > t) per the paper's descending convention.
template<class Int, class Real>
struct VreFrame {
    typename Simplex<Int>::IdxVector vertices;  // ascending
    Real compare_value;                         // see comment above
    Real diameter;                              // the filtration value
    VREdge<Int> cached_edge;                    // (s, t), s > t
};

// Run body(t, lo, hi) over `nt` contiguous chunks of the index range [0, n).
// For nt <= 1 the body runs inline on the calling thread (no thread spawned),
// so the single-threaded path pays no threading overhead. Workers must NOT
// throw on interrupt (an uncaught exception in a std::thread calls
// std::terminate); instead they poll oineus::interrupted() and return early,
// and the caller checks oineus::interrupted() after this returns.
template<class Int, class Body>
void parallel_chunks(Int n, int nt, Body&& body)
{
    if (nt <= 1) {
        body(0, Int(0), n);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const Int lo = static_cast<Int>((static_cast<long long>(n) * t) / nt);
        const Int hi = static_cast<Int>((static_cast<long long>(n) * (t + 1)) / nt);
        workers.emplace_back([&body, t, lo, hi]() { body(t, lo, hi); });
    }
    for (auto& w : workers)
        w.join();
}

// Generate every valid d-cofacet of `parent` via Cases I, II, III of VRE.
// `compare_dist(a, b)` returns the same monotonic-in-true-distance
// quantity stored in `parent.compare_value`. `emit(VreFrame&&)` is
// called for each generated cofacet.
//
// Candidate vertices are drawn from `nbrs[v_top]`, the sorted-ascending list
// of v_top's <= max_compare neighbors: every vertex of a valid cofacet is a
// <= alpha (hence <= max_compare) neighbor of v_top, so it is present in that
// list. Note `nbrs` is built at the *global* max_compare cutoff while each
// case tests against the parent's own diameter `alpha <= max_compare`, so the
// list is a superset of the true candidates -- the per-case alpha checks below
// (unchanged from the brute-force version) filter it down. The win is purely
// in scanning a neighbor list instead of the whole vertex tail.
//
// The three cases correspond to where the new vertex v slots into the
// ascending vertices_ vector; each case's index window is a binary-searched
// sub-range of the sorted neighbor list. After insertion vertices_ remains
// sorted ascending, so we pass it straight to the presorted Simplex<Int> ctor.
template<class Int, class Real, class CompareDistFn, class EmitFn>
void generate_cofacets(const VreFrame<Int, Real>& parent,
                       const std::vector<typename Simplex<Int>::IdxVector>& nbrs,
                       const CompareDistFn& compare_dist,
                       EmitFn&& emit)
{
    const auto& V = parent.vertices;
    const Real alpha = parent.compare_value;
    const Real diameter = parent.diameter;
    const Int d = static_cast<Int>(V.size());  // parent has d vertices, dim = d-1
    const Int v_top = V.back();                // paper's v_{d-1}
    const auto& nbr = nbrs[v_top];             // candidate pool, sorted ascending

    // -------- Case I: append v > v_top with d(v, V[j]) <= alpha for all j.
    for (auto it = std::upper_bound(nbr.begin(), nbr.end(), v_top);
         it != nbr.end(); ++it) {
        const Int v = *it;
        bool ok = true;
        for (Int j = 0; j < d; ++j) {
            if (compare_dist(v, V[j]) > alpha) { ok = false; break; }
        }
        if (!ok) continue;
        typename Simplex<Int>::IdxVector vs = V;
        vs.push_back(v);
        emit(VreFrame<Int, Real>{std::move(vs), alpha, diameter, parent.cached_edge});
    }

    // -------- Case II: insert v with V[d-2] < v < v_top.
    // Enabled iff parent's cached_edge.x == v_top (all full-length edges
    // are incident to v_top).
    // Conditions:
    //   d(v, v_top) <= alpha  (non-strict, since we're allowed to pair with v_top)
    //   d(v, V[j]) <  alpha   (strict)  for j < d-1
    if (d >= 2 && parent.cached_edge.x == v_top) {
        const Int v_lo = V[d - 2];
        const auto lo = std::upper_bound(nbr.begin(), nbr.end(), v_lo);
        const auto hi = std::lower_bound(nbr.begin(), nbr.end(), v_top);
        for (auto it = lo; it != hi; ++it) {
            const Int v = *it;
            if (compare_dist(v, v_top) > alpha) continue;  // <= alpha
            bool ok = true;
            for (Int j = 0; j < d - 1; ++j) {
                if (compare_dist(v, V[j]) >= alpha) { ok = false; break; }  // < alpha
            }
            if (!ok) continue;
            typename Simplex<Int>::IdxVector vs;
            vs.reserve(d + 1);
            vs.insert(vs.end(), V.begin(), V.end() - 1);  // V[0..d-2)
            vs.push_back(v);
            vs.push_back(v_top);
            emit(VreFrame<Int, Real>{std::move(vs), alpha, diameter, parent.cached_edge});
        }
    }

    // -------- Case III: insert v with V[d-3] < v < V[d-2].
    // Enabled iff parent's cached_edge == [v_top, V[d-2]] (the unique
    // full-length edge of the parent).
    // Condition: d(v, V[j]) < alpha (strict) for all j.
    //
    // Special case for d == 2 (parent is an edge): there is no V[d-3], so
    // the lower bound is -infinity (we iterate v in [0, V[d-2])).
    if (d >= 2 && parent.cached_edge.x == v_top && parent.cached_edge.y == V[d - 2]) {
        const Int v_hi = V[d - 2];
        const Int v_lo_excl = (d >= 3) ? V[d - 3] : Int(-1);
        const auto lo = std::upper_bound(nbr.begin(), nbr.end(), v_lo_excl);
        const auto hi = std::lower_bound(nbr.begin(), nbr.end(), v_hi);
        for (auto it = lo; it != hi; ++it) {
            const Int v = *it;
            bool ok = true;
            for (Int j = 0; j < d; ++j) {
                if (compare_dist(v, V[j]) >= alpha) { ok = false; break; }  // < alpha
            }
            if (!ok) continue;
            typename Simplex<Int>::IdxVector vs;
            vs.reserve(d + 1);
            // Take everything below V[d-2], append v, then V[d-2..d-1].
            // For d == 2 this is empty + [v] + [V[0], V[1]].
            if (d >= 3)
                vs.insert(vs.end(), V.begin(), V.end() - 2);
            vs.push_back(v);
            vs.insert(vs.end(), V.end() - 2, V.end());
            emit(VreFrame<Int, Real>{std::move(vs), alpha, diameter, parent.cached_edge});
        }
    }
}

// Templated VRE driver. Builds the full filtration up to `max_dim` and
// (optionally) the parallel critical-edge array.
//
// `compare_dist(a, b) -> Real` is the cutoff/comparison value (any
// quantity monotonic in true distance). `compare_to_diameter(c) -> Real`
// converts a compare-value to the raw filtration value to store on the
// cell. `max_compare` is the cutoff threshold expressed in compare units.
// The two natural choices are:
//
//   - points path: compare_dist = squared Euclidean distance,
//                  compare_to_diameter = sqrt,
//                  max_compare = max_diameter * max_diameter.
//
//   - dist-matrix path: compare_dist = the user's raw distance (passed
//                  through unchanged), compare_to_diameter = identity,
//                  max_compare = max_diameter. This preserves the
//                  user's distance bit-for-bit and avoids the
//                  d-times-d overflow risk on inputs near
//                  std::numeric_limits<Real>::max().
//
// On output, `simplices` is in (dim, value) sorted order: vertices in
// vertex-id order, edges in increasing-distance order, and each higher
// layer in the order produced by walking the previous layer in order. The
// caller may pass `simplices` straight to the presorted Filtration
// constructor (`Filtration(presorted, ...)`) -- no global sort is needed.
//
// `edges` is parallel to `simplices`: entry i is the cached lex-first
// full-length edge of cells_[i] (or a (v, v) self-loop for vertex i).
// Because cells go in already in sorted order, `edges` does not need
// post-sort permutation either.
// UnderCell is the stored simplicial cell type: Simplex<Int> (Fat) for the classic
// fat filtration, or Simplex<Int, BitPacked<Int,Word>> for the bit-packed one. The
// enumeration is identical either way -- it always works on ascending vertex lists --
// only how each cell is STORED differs, decided by `if constexpr` on the encoding in
// the record/add_cell sinks. `bits` is the PackedGeom field width for the packed
// encoding (ignored for Fat).
template<class Int, class Real, class UnderCell, class CompareDistFn, class CompareToDiamFn>
void vre_build(Int n_points,
               dim_type max_dim,
               Real max_compare,
               const CompareDistFn& compare_dist,
               const CompareToDiamFn& compare_to_diameter,
               bool collect_edges,
               int bits,
               typename Filtration<UnderCell, Real>::CellVector& simplices,
               std::vector<VREdge<Int>>& edges,
               int n_threads = 1)
{
    using Cell = CellWithValue<UnderCell, Real>;
    using Enc = typename UnderCell::Enc;
    // Per-layer frame buffers hold the intermediate-dimension simplices (the
    // bulk for max_dim >= 3); route their growth through jemalloc too.
    using FrameVec = std::vector<VreFrame<Int, Real>, JeAllocator<VreFrame<Int, Real>>>;

    // Construct the stored cell from an ascending vertex list. Fat uses the presorted
    // Simplex ctor (move, no re-sort); a packed encoding packs the ascending ids into
    // a word with `bits` per field. `vs` is ascending by construction (generate_cofacets).
    auto make_under = [bits](typename Simplex<Int>::IdxVector&& vs) -> UnderCell {
        if constexpr (std::is_same_v<Enc, Fat<Int>>)
            return UnderCell(presorted, std::move(vs));
        else
            return UnderCell(Enc(vs, bits));
    };

    auto record = [&](typename Simplex<Int>::IdxVector vs, Real diam, VREdge<Int> e) {
        simplices.emplace_back(make_under(std::move(vs)), diam);
        if (collect_edges) edges.push_back(e);
    };

    // ---- Layer 0: vertices (all value 0, in vertex-id order).
    for (Int v = 0; v < n_points; ++v) {
        record({v}, Real(0), VREdge<Int>{v, v});
    }

    if (max_dim < 1)
        return;

    // nbrs[v] = sorted-ascending list of v's <= max_compare neighbors. Built
    // only when higher layers will consume it (max_dim >= 2); it lets
    // generate_cofacets scan a neighbor list instead of the whole vertex tail.
    std::vector<typename Simplex<Int>::IdxVector> nbrs;
    const bool build_nbrs = (max_dim >= 2);
    if (build_nbrs)
        nbrs.resize(static_cast<size_t>(n_points));

    // ---- Layer 1: collect surviving edges (and the adjacency), then sort
    // once by compare-value. This is the *only* sort the algorithm needs:
    // every higher layer inherits its parent's diameter exactly, so processing
    // the sorted layer-1 frames in order yields all subsequent layers already
    // in increasing-value order.
    //
    // No reserve(n*(n-1)/2): for sparse inputs that would force O(n^2)
    // allocation up front even when few edges survive. std::vector's geometric
    // growth is fine.
    FrameVec current;
    {
        struct EdgeData {
            Int u;
            Int v;
            Real cv;
        };
        std::vector<EdgeData> raw_edges;

        if (n_threads <= 1) {
            // Sequential triangular scan. Each surviving edge (u,v) feeds
            // raw_edges once and the adjacency at both endpoints; the
            // for-u / for-v>u order makes each nbrs[w] come out ascending with
            // no explicit sort (its <w neighbors are appended in increasing-u
            // order, then its >w neighbors in increasing order).
            for (Int u = 0; u < n_points; ++u) {
                if ((u & 1023) == 0 && oineus::interrupted())
                    throw oineus::interrupted_exception{};
                for (Int v = u + 1; v < n_points; ++v) {
                    const Real cv = compare_dist(u, v);
                    if (cv > max_compare) continue;
                    raw_edges.push_back({u, v, cv});
                    if (build_nbrs) { nbrs[u].push_back(v); nbrs[v].push_back(u); }
                }
            }
        } else {
            // Parallel over u-rows. Each worker owns a disjoint block of u, so
            // it writes only its own nbrs[u] rows (race-free; nbrs is pre-sized
            // so no reallocation) and a thread-local edge buffer. When building
            // the adjacency a worker scans every v != u -- computing each pair
            // distance twice overall, the price of avoiding synchronization,
            // which also keeps each nbrs[u] ascending; otherwise it scans only
            // v > u. Buffers are concatenated in u-order afterwards (the final
            // std::sort makes collection order irrelevant anyway).
            const int nt = std::min<int>(n_threads,
                    static_cast<int>(std::max<Int>(Int(1), n_points)));
            std::vector<std::vector<EdgeData>> local_edges(static_cast<size_t>(nt));
            parallel_chunks<Int>(n_points, nt, [&](int t, Int lo, Int hi) {
                auto& le = local_edges[static_cast<size_t>(t)];
                for (Int u = lo; u < hi; ++u) {
                    if ((u & 1023) == 0 && oineus::interrupted())
                        return;
                    if (build_nbrs) {
                        auto& nu = nbrs[static_cast<size_t>(u)];
                        for (Int v = 0; v < n_points; ++v) {
                            if (v == u) continue;
                            const Real cv = compare_dist(u, v);
                            if (cv > max_compare) continue;
                            nu.push_back(v);
                            if (v > u) le.push_back({u, v, cv});
                        }
                    } else {
                        for (Int v = u + 1; v < n_points; ++v) {
                            const Real cv = compare_dist(u, v);
                            if (cv > max_compare) continue;
                            le.push_back({u, v, cv});
                        }
                    }
                }
            });
            if (oineus::interrupted())
                throw oineus::interrupted_exception{};
            size_t total = 0;
            for (const auto& le : local_edges) total += le.size();
            raw_edges.reserve(total);
            for (auto& le : local_edges)
                raw_edges.insert(raw_edges.end(), le.begin(), le.end());
        }

        // Sort by (cv, u, v): primary key is the compare-value, ties go to lex
        // (u, v). The explicit (u, v) tiebreaker lets us use std::sort
        // (in-place introsort) instead of std::stable_sort, which allocates an
        // O(E) auxiliary buffer for the merge phase.
        std::sort(raw_edges.begin(), raw_edges.end(),
                [](const EdgeData& a, const EdgeData& b) {
                    return std::tie(a.cv, a.u, a.v) < std::tie(b.cv, b.u, b.v);
                });

        current.reserve(raw_edges.size());
        for (const auto& ed : raw_edges) {
            const Real diam = compare_to_diameter(ed.cv);
            const VREdge<Int> e{ed.v, ed.u};   // s = v > u = t (paper notation)
            current.push_back({{ed.u, ed.v}, ed.cv, diam, e});
            record({ed.u, ed.v}, diam, e);
        }
    }

    // ---- Layers 2 .. max_dim. `current` is in increasing-value order at every
    // step; cofacets share the parent's value exactly, so each `next` is also
    // in increasing-value order without sorting.
    //
    // The final layer (d == max_dim) is special: we don't build `next`, and the
    // emitter moves child.vertices straight into the Simplex record.
    //
    // Parallel path (n_threads > 1): split `current` into contiguous chunks,
    // each worker emits into thread-local buffers, then concatenate in chunk
    // order. Because chunks are contiguous ranges of the already-sorted
    // `current` and concatenation preserves chunk order, the global (dim,value)
    // order the presorted Filtration needs is preserved (cofacets of one parent
    // all share its diameter, so their intra-parent order is free). `nbrs` and
    // `compare_dist` are read-only here, hence safe to share across workers.
    for (dim_type d = 2; d <= max_dim; ++d) {
        const bool is_final = (d == max_dim);

        if (n_threads <= 1) {
            if (is_final) {
                auto emit_final = [&](VreFrame<Int, Real>&& child) {
                    record(std::move(child.vertices), child.diameter, child.cached_edge);
                };
                size_t i = 0;
                for (const auto& parent : current) {
                    if ((i & 4095) == 0 && oineus::interrupted())
                        throw oineus::interrupted_exception{};
                    ++i;
                    generate_cofacets<Int, Real>(parent, nbrs, compare_dist, emit_final);
                }
                // Loop terminates next iteration; no need to update `current`.
            } else {
                FrameVec next;
                auto emit_frame = [&](VreFrame<Int, Real>&& child) {
                    record(child.vertices, child.diameter, child.cached_edge);
                    next.push_back(std::move(child));
                };
                size_t i = 0;
                for (const auto& parent : current) {
                    if ((i & 4095) == 0 && oineus::interrupted())
                        throw oineus::interrupted_exception{};
                    ++i;
                    generate_cofacets<Int, Real>(parent, nbrs, compare_dist, emit_frame);
                }
                current = std::move(next);
                if (current.empty()) break;  // no cofacets here, none higher up
            }
        } else {
            const int nt = std::min<int>(n_threads,
                    static_cast<int>(std::max<size_t>(size_t(1), current.size())));
            std::vector<std::vector<Cell, JeAllocator<Cell>>> loc_cells(static_cast<size_t>(nt));
            std::vector<std::vector<VREdge<Int>>> loc_edges(static_cast<size_t>(nt));
            std::vector<FrameVec> loc_next(
                    is_final ? size_t(0) : static_cast<size_t>(nt));

            parallel_chunks<Int>(static_cast<Int>(current.size()), nt,
                    [&](int t, Int lo, Int hi) {
                auto& cells = loc_cells[static_cast<size_t>(t)];
                auto& ce = loc_edges[static_cast<size_t>(t)];
                auto add_cell = [&](typename Simplex<Int>::IdxVector vs, Real diam,
                                    VREdge<Int> e) {
                    cells.emplace_back(make_under(std::move(vs)), diam);
                    if (collect_edges) ce.push_back(e);
                };
                if (is_final) {
                    auto emit_final = [&](VreFrame<Int, Real>&& child) {
                        add_cell(std::move(child.vertices), child.diameter, child.cached_edge);
                    };
                    for (Int i = lo; i < hi; ++i) {
                        if (((i - lo) & 4095) == 0 && oineus::interrupted())
                            return;
                        generate_cofacets<Int, Real>(current[static_cast<size_t>(i)],
                                nbrs, compare_dist, emit_final);
                    }
                } else {
                    auto& nx = loc_next[static_cast<size_t>(t)];
                    auto emit_frame = [&](VreFrame<Int, Real>&& child) {
                        add_cell(child.vertices, child.diameter, child.cached_edge);
                        nx.push_back(std::move(child));
                    };
                    for (Int i = lo; i < hi; ++i) {
                        if (((i - lo) & 4095) == 0 && oineus::interrupted())
                            return;
                        generate_cofacets<Int, Real>(current[static_cast<size_t>(i)],
                                nbrs, compare_dist, emit_frame);
                    }
                }
            });
            if (oineus::interrupted())
                throw oineus::interrupted_exception{};

            // Concatenate thread-local buffers in chunk order (preserves the
            // global sorted order). Free each chunk as we drain it to keep peak
            // memory near one layer rather than two.
            size_t tot_cells = 0, tot_next = 0;
            for (int t = 0; t < nt; ++t) {
                tot_cells += loc_cells[static_cast<size_t>(t)].size();
                if (!is_final) tot_next += loc_next[static_cast<size_t>(t)].size();
            }
            simplices.reserve(simplices.size() + tot_cells);
            if (collect_edges) edges.reserve(edges.size() + tot_cells);
            FrameVec next;
            next.reserve(tot_next);
            for (int t = 0; t < nt; ++t) {
                auto& cells = loc_cells[static_cast<size_t>(t)];
                auto& ce = loc_edges[static_cast<size_t>(t)];
                for (auto& c : cells) simplices.push_back(std::move(c));
                if (collect_edges)
                    for (const auto& e : ce) edges.push_back(e);
                cells.clear(); cells.shrink_to_fit();
                ce.clear(); ce.shrink_to_fit();
                if (!is_final) {
                    auto& nxt = loc_next[static_cast<size_t>(t)];
                    for (auto& f : nxt) next.push_back(std::move(f));
                    nxt.clear(); nxt.shrink_to_fit();
                }
            }
            if (!is_final) {
                current = std::move(next);
                if (current.empty()) break;  // no cofacets here, none higher up
            }
        }
    }
}

} // namespace detail

// ============================================================================
//                        Public entry points
// ============================================================================

// All four entry points hand the resulting cells (already in (dim, value)
// order, by construction in `vre_build`) to the presorted Filtration
// constructor, which skips the global sort and only does the sequential
// uid_to_sorted_id population. `n_threads` (default 1) is forwarded to
// `vre_build`, which parallelizes the all-pairs edge/neighbor scan and the
// per-layer cofacet generation; the presorted ctor itself has no parallel
// phase.

// Helpers used by the four entry points: the points-cloud path uses
// squared Euclidean distance as its compare unit (and sqrt to recover
// the diameter); the dist-matrix path passes raw distances through
// unchanged (compare unit == diameter, identity conversion). The
// dist-matrix path therefore preserves the user-supplied distance
// bit-for-bit and avoids any d*d overflow on inputs near the FP
// dynamic range.

// Points + filtration only (no critical edges).
template<class Int, class Real, std::size_t D>
auto get_vr_filtration_inorder(const std::vector<Point<Real, D>>& points,
                               dim_type max_dim = D,
                               Real max_diameter = std::numeric_limits<Real>::max(),
                               int n_threads = 1)
    -> Filtration<Simplex<Int>, Real>
{
    using Cell = CellWithValue<Simplex<Int>, Real>;  // used by vector type below
    auto compare_dist = [&](Int a, Int b) -> Real {
        return sq_dist<Real, D>(points[a], points[b]);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return std::sqrt(cv); };
    const Real max_compare = (max_diameter >= std::numeric_limits<Real>::max() / 2)
        ? std::numeric_limits<Real>::max()
        : max_diameter * max_diameter;

    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;  // unused
    detail::vre_build<Int, Real, Simplex<Int>>(static_cast<Int>(points.size()), max_dim,
                                 max_compare, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/false, /*bits=*/0, simplices, edges, n_threads);
    auto fil = Filtration<Simplex<Int>, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    return fil;
}

// Distance matrix + filtration only (no critical edges).
template<class Int, class Real>
auto get_vr_filtration_inorder(const DistMatrix<Real>& dm,
                               dim_type max_dim,
                               Real max_diameter = std::numeric_limits<Real>::max(),
                               int n_threads = 1)
    -> Filtration<Simplex<Int>, Real>
{
    using Cell = CellWithValue<Simplex<Int>, Real>;  // used by vector type below
    auto compare_dist = [&](Int a, Int b) -> Real {
        return dm.get_distance(a, b);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return cv; };

    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;  // unused
    detail::vre_build<Int, Real, Simplex<Int>>(static_cast<Int>(dm.n_points), max_dim,
                                 max_diameter, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/false, /*bits=*/0, simplices, edges, n_threads);
    auto fil = Filtration<Simplex<Int>, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    return fil;
}

// Points + critical edges. `edges` is parallel to the simplices vector
// (entry-by-entry); since VRE delivers simplices in already-sorted order,
// no post-sort permutation is needed.
template<class Int, class Real, std::size_t D>
auto get_vr_filtration_and_critical_edges_inorder(
        const std::vector<Point<Real, D>>& points,
        dim_type max_dim = D,
        Real max_diameter = std::numeric_limits<Real>::max(),
        int n_threads = 1)
    -> std::pair<Filtration<Simplex<Int>, Real>, std::vector<VREdge<Int>>>
{
    using Cell = CellWithValue<Simplex<Int>, Real>;  // used by vector type below
    auto compare_dist = [&](Int a, Int b) -> Real {
        return sq_dist<Real, D>(points[a], points[b]);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return std::sqrt(cv); };
    const Real max_compare = (max_diameter >= std::numeric_limits<Real>::max() / 2)
        ? std::numeric_limits<Real>::max()
        : max_diameter * max_diameter;

    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;
    detail::vre_build<Int, Real, Simplex<Int>>(static_cast<Int>(points.size()), max_dim,
                                 max_compare, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/true, /*bits=*/0, simplices, edges, n_threads);

    auto fil = Filtration<Simplex<Int>, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    return std::make_pair(std::move(fil), std::move(edges));
}

// Distance matrix + critical edges.
template<class Int, class Real>
auto get_vr_filtration_and_critical_edges_inorder(
        const DistMatrix<Real>& dm,
        dim_type max_dim,
        Real max_diameter = std::numeric_limits<Real>::max(),
        int n_threads = 1)
    -> std::pair<Filtration<Simplex<Int>, Real>, std::vector<VREdge<Int>>>
{
    using Cell = CellWithValue<Simplex<Int>, Real>;  // used by vector type below
    auto compare_dist = [&](Int a, Int b) -> Real {
        return dm.get_distance(a, b);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return cv; };

    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;
    detail::vre_build<Int, Real, Simplex<Int>>(static_cast<Int>(dm.n_points), max_dim,
                                 max_diameter, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/true, /*bits=*/0, simplices, edges, n_threads);

    auto fil = Filtration<Simplex<Int>, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    return std::make_pair(std::move(fil), std::move(edges));
}

// ---- Packed (bit-packed) VR builders. Same VRE enumeration, but each stored cell is
// a Simplex<Int, BitPacked<Int,Word>> whose sorted vertex ids are packed into one Word
// (bits = ceil(log2(n_points)) per field); the shared PackedGeom{bits} is set on the
// filtration so the (co)boundary can unpack. The caller picks Word via
// bit_packing_fits<Word>(n_points, max_dim); reduce/diagram are identical to the fat
// path (BitPacked is HasDirectCoboundary=false -> antitranspose, like the fat Simplex).
// The critical-edge VREdge array is vertex-id pairs, encoding-independent, and stays
// aligned to the presorted cells exactly as in the fat path.
template<class Int, class Real, class Word, std::size_t D>
auto get_vr_filtration_packed_inorder(const std::vector<Point<Real, D>>& points,
                                      dim_type max_dim = D,
                                      Real max_diameter = std::numeric_limits<Real>::max(),
                                      int n_threads = 1)
    -> Filtration<Simplex<Int, BitPacked<Int, Word>>, Real>
{
    using PackedCell = Simplex<Int, BitPacked<Int, Word>>;
    using Cell = CellWithValue<PackedCell, Real>;
    auto compare_dist = [&](Int a, Int b) -> Real {
        return sq_dist<Real, D>(points[a], points[b]);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return std::sqrt(cv); };
    const Real max_compare = (max_diameter >= std::numeric_limits<Real>::max() / 2)
        ? std::numeric_limits<Real>::max()
        : max_diameter * max_diameter;

    const int bits = packed_vertex_bits(static_cast<size_t>(points.size()));
    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;  // unused
    detail::vre_build<Int, Real, PackedCell>(static_cast<Int>(points.size()), max_dim,
                                 max_compare, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/false, bits, simplices, edges, n_threads);
    auto fil = Filtration<PackedCell, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    fil.set_geometry(PackedGeom{bits});
    return fil;
}

template<class Int, class Real, class Word>
auto get_vr_filtration_packed_inorder(const DistMatrix<Real>& dm,
                                      dim_type max_dim,
                                      Real max_diameter = std::numeric_limits<Real>::max(),
                                      int n_threads = 1)
    -> Filtration<Simplex<Int, BitPacked<Int, Word>>, Real>
{
    using PackedCell = Simplex<Int, BitPacked<Int, Word>>;
    using Cell = CellWithValue<PackedCell, Real>;
    auto compare_dist = [&](Int a, Int b) -> Real {
        return dm.get_distance(a, b);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return cv; };

    const int bits = packed_vertex_bits(static_cast<size_t>(dm.n_points));
    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;  // unused
    detail::vre_build<Int, Real, PackedCell>(static_cast<Int>(dm.n_points), max_dim,
                                 max_diameter, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/false, bits, simplices, edges, n_threads);
    auto fil = Filtration<PackedCell, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    fil.set_geometry(PackedGeom{bits});
    return fil;
}

template<class Int, class Real, class Word, std::size_t D>
auto get_vr_filtration_and_critical_edges_packed_inorder(
        const std::vector<Point<Real, D>>& points,
        dim_type max_dim = D,
        Real max_diameter = std::numeric_limits<Real>::max(),
        int n_threads = 1)
    -> std::pair<Filtration<Simplex<Int, BitPacked<Int, Word>>, Real>, std::vector<VREdge<Int>>>
{
    using PackedCell = Simplex<Int, BitPacked<Int, Word>>;
    using Cell = CellWithValue<PackedCell, Real>;
    auto compare_dist = [&](Int a, Int b) -> Real {
        return sq_dist<Real, D>(points[a], points[b]);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return std::sqrt(cv); };
    const Real max_compare = (max_diameter >= std::numeric_limits<Real>::max() / 2)
        ? std::numeric_limits<Real>::max()
        : max_diameter * max_diameter;

    const int bits = packed_vertex_bits(static_cast<size_t>(points.size()));
    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;
    detail::vre_build<Int, Real, PackedCell>(static_cast<Int>(points.size()), max_dim,
                                 max_compare, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/true, bits, simplices, edges, n_threads);
    auto fil = Filtration<PackedCell, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    fil.set_geometry(PackedGeom{bits});
    return std::make_pair(std::move(fil), std::move(edges));
}

template<class Int, class Real, class Word>
auto get_vr_filtration_and_critical_edges_packed_inorder(
        const DistMatrix<Real>& dm,
        dim_type max_dim,
        Real max_diameter = std::numeric_limits<Real>::max(),
        int n_threads = 1)
    -> std::pair<Filtration<Simplex<Int, BitPacked<Int, Word>>, Real>, std::vector<VREdge<Int>>>
{
    using PackedCell = Simplex<Int, BitPacked<Int, Word>>;
    using Cell = CellWithValue<PackedCell, Real>;
    auto compare_dist = [&](Int a, Int b) -> Real {
        return dm.get_distance(a, b);
    };
    auto compare_to_diameter = [](Real cv) -> Real { return cv; };

    const int bits = packed_vertex_bits(static_cast<size_t>(dm.n_points));
    std::vector<Cell, JeAllocator<Cell>> simplices;
    std::vector<VREdge<Int>> edges;
    detail::vre_build<Int, Real, PackedCell>(static_cast<Int>(dm.n_points), max_dim,
                                 max_diameter, compare_dist, compare_to_diameter,
                                 /*collect_edges=*/true, bits, simplices, edges, n_threads);
    auto fil = Filtration<PackedCell, Real>(presorted, std::move(simplices), /*negate=*/false);
    fil.set_kind(FiltrationKind::Vr);
    fil.set_geometry(PackedGeom{bits});
    return std::make_pair(std::move(fil), std::move(edges));
}

} // namespace oineus

#endif // OINEUS_VIETORIS_RIPS_INORDER_H
