#ifndef OINEUS_VIETORIS_RIPS_INORDER_H
#define OINEUS_VIETORIS_RIPS_INORDER_H

#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <cassert>
#include <utility>

#include "filtration.h"
#include "vietoris_rips.h"

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
// We carry the squared diameter (alpha^2) so the inner-loop distance
// comparison stays in squared-distance units (the closure returns
// d^2). The cached_edge stores the lex-first full-length edge of the
// simplex, written (x = s, y = t) with s > t (the paper's
// convention).
template<class Int, class Real>
struct VreFrame {
    typename Simplex<Int>::IdxVector vertices;  // ascending
    Real diameter_sq;                           // alpha^2
    VREdge<Int> cached_edge;                    // (s, t), s > t
};

// Generate every valid d-cofacet of `parent` via Cases I, II, III of VRE.
// `sq_dist(a, b)` must return d(a, b)^2.
// `emit(VreFrame<Int, Real>&&)` is called for each generated cofacet.
//
// The three cases correspond to where the new vertex v slots into the
// ascending vertices_ vector. After insertion vertices_ remains sorted
// ascending, so we can pass it directly to the Simplex<Int> ctor without
// re-sorting (the ctor will re-sort anyway, harmlessly).
template<class Int, class Real, class SqDistFn, class EmitFn>
void generate_cofacets(const VreFrame<Int, Real>& parent,
                       Int n_points,
                       const SqDistFn& sq_dist,
                       EmitFn&& emit)
{
    const auto& V = parent.vertices;
    const Real alpha_sq = parent.diameter_sq;
    const Int d = static_cast<Int>(V.size());  // parent has d vertices, dim = d-1
    const Int v_top = V.back();                // paper's v_{d-1}

    // -------- Case I: append v > v_top with d(v, V[j]) <= alpha for all j.
    // In squared units: sq_dist(v, V[j]) <= alpha_sq.
    for (Int v = v_top + 1; v < n_points; ++v) {
        bool ok = true;
        for (Int j = 0; j < d; ++j) {
            if (sq_dist(v, V[j]) > alpha_sq) { ok = false; break; }
        }
        if (!ok) continue;
        typename Simplex<Int>::IdxVector vs = V;
        vs.push_back(v);
        emit(VreFrame<Int, Real>{std::move(vs), alpha_sq, parent.cached_edge});
    }

    // -------- Case II: insert v with V[d-2] < v < v_top.
    // Enabled iff parent's cached_edge.x == v_top (all full-length edges
    // are incident to v_top).
    // Conditions:
    //   d(v, v_top) <= alpha  (non-strict, since we're allowed to pair with v_top)
    //   d(v, V[j]) <  alpha   (strict)  for j < d-1
    if (d >= 2 && parent.cached_edge.x == v_top) {
        const Int v_lo = V[d - 2];
        for (Int v = v_lo + 1; v < v_top; ++v) {
            if (sq_dist(v, v_top) > alpha_sq) continue;  // <= alpha
            bool ok = true;
            for (Int j = 0; j < d - 1; ++j) {
                if (sq_dist(v, V[j]) >= alpha_sq) { ok = false; break; }  // < alpha
            }
            if (!ok) continue;
            typename Simplex<Int>::IdxVector vs;
            vs.reserve(d + 1);
            vs.insert(vs.end(), V.begin(), V.end() - 1);  // V[0..d-2)
            vs.push_back(v);
            vs.push_back(v_top);
            emit(VreFrame<Int, Real>{std::move(vs), alpha_sq, parent.cached_edge});
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
        for (Int v = v_lo_excl + 1; v < v_hi; ++v) {
            bool ok = true;
            for (Int j = 0; j < d; ++j) {
                if (sq_dist(v, V[j]) >= alpha_sq) { ok = false; break; }  // < alpha
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
            emit(VreFrame<Int, Real>{std::move(vs), alpha_sq, parent.cached_edge});
        }
    }
}

// Templated VRE driver. Builds the full filtration up to `max_dim` and
// (optionally) the parallel critical-edge array. The caller supplies a
// `sq_dist(Int, Int) -> Real` closure.
//
// On output, `simplices` is the unsorted vector ready for the Filtration
// constructor (which performs the parallel sort), and `edges` is parallel
// to `simplices` (entry-by-entry) with the cached lex-first full-length
// edge (or a self-loop for vertices). Caller is responsible for the
// post-sort permutation.
template<class Int, class Real, class SqDistFn>
void vre_build(Int n_points,
               dim_type max_dim,
               Real max_diameter,
               const SqDistFn& sq_dist,
               bool collect_edges,
               std::vector<CellWithValue<Simplex<Int>, Real>>& simplices,
               std::vector<VREdge<Int>>& edges)
{
    const Real max_diam_sq = (max_diameter >= std::numeric_limits<Real>::max() / 2)
        ? std::numeric_limits<Real>::max()
        : max_diameter * max_diameter;

    auto record = [&](typename Simplex<Int>::IdxVector vs, Real diam, VREdge<Int> e) {
        simplices.emplace_back(Simplex<Int>(std::move(vs)), diam);
        if (collect_edges) edges.push_back(e);
    };

    // ---- Layer 0: vertices.
    for (Int v = 0; v < n_points; ++v) {
        record({v}, Real(0), VREdge<Int>{v, v});
    }

    if (max_dim < 1)
        return;

    // ---- Layer 1: edges. Each edge is its own cached lex-first full-length edge.
    std::vector<VreFrame<Int, Real>> current;
    for (Int u = 0; u < n_points; ++u) {
        for (Int v = u + 1; v < n_points; ++v) {
            const Real dsq = sq_dist(u, v);
            if (dsq > max_diam_sq) continue;
            const Real diam = std::sqrt(dsq);
            const VREdge<Int> e{v, u};  // s = v > u = t
            current.push_back({{u, v}, dsq, e});
            record({u, v}, diam, e);
        }
    }

    // ---- Layers 2 .. max_dim
    for (dim_type d = 2; d <= max_dim; ++d) {
        std::vector<VreFrame<Int, Real>> next;
        auto emit_frame = [&](VreFrame<Int, Real>&& child) {
            const Real diam = std::sqrt(child.diameter_sq);
            record(child.vertices, diam, child.cached_edge);
            next.push_back(std::move(child));
        };
        for (const auto& parent : current) {
            generate_cofacets<Int, Real>(parent, n_points, sq_dist, emit_frame);
        }
        current = std::move(next);
        if (current.empty()) break;  // no more cofacets possible at higher d
    }
}

} // namespace detail

// ============================================================================
//                        Public entry points
// ============================================================================

// Points + filtration only (no critical edges).
template<class Int, class Real, std::size_t D>
auto get_vr_filtration_inorder(const std::vector<Point<Real, D>>& points,
                               dim_type max_dim = D,
                               Real max_diameter = std::numeric_limits<Real>::max(),
                               int n_threads = 1)
    -> Filtration<Simplex<Int>, Real>
{
    using Cell = CellWithValue<Simplex<Int>, Real>;  // used by vector type below
    auto sq_dist_fn = [&](Int a, Int b) -> Real {
        return sq_dist<Real, D>(points[a], points[b]);
    };
    std::vector<Cell> simplices;
    std::vector<VREdge<Int>> edges;  // unused
    detail::vre_build<Int, Real>(static_cast<Int>(points.size()), max_dim,
                                 max_diameter, sq_dist_fn,
                                 /*collect_edges=*/false, simplices, edges);
    return Filtration<Simplex<Int>, Real>(std::move(simplices), /*negate=*/false, n_threads);
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
    auto sq_dist_fn = [&](Int a, Int b) -> Real {
        const Real d = dm.get_distance(a, b);
        return d * d;
    };
    std::vector<Cell> simplices;
    std::vector<VREdge<Int>> edges;  // unused
    detail::vre_build<Int, Real>(static_cast<Int>(dm.n_points), max_dim,
                                 max_diameter, sq_dist_fn,
                                 /*collect_edges=*/false, simplices, edges);
    return Filtration<Simplex<Int>, Real>(std::move(simplices), /*negate=*/false, n_threads);
}

// Points + critical edges.
template<class Int, class Real, std::size_t D>
auto get_vr_filtration_and_critical_edges_inorder(
        const std::vector<Point<Real, D>>& points,
        dim_type max_dim = D,
        Real max_diameter = std::numeric_limits<Real>::max(),
        int n_threads = 1)
    -> std::pair<Filtration<Simplex<Int>, Real>, std::vector<VREdge<Int>>>
{
    using Cell = CellWithValue<Simplex<Int>, Real>;  // used by vector type below
    auto sq_dist_fn = [&](Int a, Int b) -> Real {
        return sq_dist<Real, D>(points[a], points[b]);
    };
    std::vector<Cell> simplices;
    std::vector<VREdge<Int>> edges;
    detail::vre_build<Int, Real>(static_cast<Int>(points.size()), max_dim,
                                 max_diameter, sq_dist_fn,
                                 /*collect_edges=*/true, simplices, edges);

    auto fil = Filtration<Simplex<Int>, Real>(std::move(simplices), /*negate=*/false, n_threads);

    // Permute edges to follow the sorted filtration order (same pattern as
    // vietoris_rips.h:248-254).
    std::vector<VREdge<Int>> sorted_edges;
    sorted_edges.reserve(edges.size());
    for (size_t i = 0; i < edges.size(); ++i)
        sorted_edges.push_back(edges[fil.get_id_by_sorted_id(i)]);

    return std::make_pair(std::move(fil), std::move(sorted_edges));
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
    auto sq_dist_fn = [&](Int a, Int b) -> Real {
        const Real d = dm.get_distance(a, b);
        return d * d;
    };
    std::vector<Cell> simplices;
    std::vector<VREdge<Int>> edges;
    detail::vre_build<Int, Real>(static_cast<Int>(dm.n_points), max_dim,
                                 max_diameter, sq_dist_fn,
                                 /*collect_edges=*/true, simplices, edges);

    auto fil = Filtration<Simplex<Int>, Real>(std::move(simplices), /*negate=*/false, n_threads);

    std::vector<VREdge<Int>> sorted_edges;
    sorted_edges.reserve(edges.size());
    for (size_t i = 0; i < edges.size(); ++i)
        sorted_edges.push_back(edges[fil.get_id_by_sorted_id(i)]);

    return std::make_pair(std::move(fil), std::move(sorted_edges));
}

} // namespace oineus

#endif // OINEUS_VIETORIS_RIPS_INORDER_H
