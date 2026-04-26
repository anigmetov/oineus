/*

Copyright (c) 2015, M. Kerber, D. Morozov, A. Nigmetov
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
(Enhancements) to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to copyright holder,
without imposing a separate written license agreement for such Enhancements,
then you hereby grant the following license: a  non-exclusive, royalty-free
perpetual license to install, use, modify, prepare derivative works, incorporate
into other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

*/

#ifndef HERA_BOTTLENECK_H
#define HERA_BOTTLENECK_H


#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <array>
#include <sstream>
#include <string>
#include <utility>

#include <hera/common.h>
#include <hera/wasserstein.h>     // for hera::InfKind, kNumInfKinds
#include "bottleneck/bottleneck_detail.h"
#include "bottleneck/basic_defs_bt.h"
#include "bottleneck/bound_match.h"

namespace hera {
    // internal_p defines cost function on edges (use hera::get_infinity(),
    // if you want to explicitly refer to l_inf, but that's default value
    // delta is relative error, default is 1 percent
    template<class Real = double>
    struct BottleneckParams
    {
        Real internal_p { hera::get_infinity() };
        Real delta { 0.01 };
    };

    // functions taking containers as input
    // template parameter PairContainer must be a container of pairs of real
    // numbers (pair.first = x-coordinate, pair.second = y-coordinate)
    // PairContainer class must support iteration of the form
    // for(it = pairContainer.begin(); it != pairContainer.end(); ++it)

    // all functions in this header are wrappers around
    // functions from hera::bt namespace

    // get exact bottleneck distance,
    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistExact(PairContainer& dgm_A, PairContainer& dgm_B, const int decPrecision,
                        hera::bt::MatchingEdge<typename DiagramTraits<PairContainer>::RealType>& longest_edge,
                        bool compute_longest_edge = true)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(dgm_A);
        hera::bt::DiagramPointSet<Real> b(dgm_B);
        return hera::bt::bottleneckDistExact(a, b, decPrecision, longest_edge, compute_longest_edge);
    }

    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistExact(PairContainer& dgm_A, PairContainer& dgm_B, const int decPrecision)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::MatchingEdge<Real> longest_edge;
        return bottleneckDistExact(dgm_A, dgm_B, decPrecision, longest_edge, false);
    }


    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistExact(PairContainer& dgm_A, PairContainer& dgm_B)
    {
        int dec_precision = 14;
        return bottleneckDistExact(dgm_A, dgm_B, dec_precision);
    }


// return the interval (distMin, distMax) such that:
// a) actual bottleneck distance between A and B is contained in the interval
// b) if the interval is not (0,0), then  (distMax - distMin) / distMin < delta
    template<class PairContainer>
    std::pair<typename DiagramTraits<PairContainer>::RealType, typename DiagramTraits<PairContainer>::RealType>
    bottleneckDistApproxInterval(PairContainer& dgm_A, PairContainer& dgm_B,
                                 const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(dgm_A);
        hera::bt::DiagramPointSet<Real> b(dgm_B);
        return hera::bt::bottleneckDistApproxInterval(a, b, delta);
    }

// use sampling heuristic: discard most of the points with small persistency
// to get a good initial approximation of the bottleneck distance
    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistApproxHeur(PairContainer& dgm_A, PairContainer& dgm_B,
                             const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(dgm_A);
        hera::bt::DiagramPointSet<Real> b(dgm_B);
        std::pair<Real, Real> resPair = hera::bt::bottleneckDistApproxIntervalHeur(a, b, delta);
        return resPair.second;
    }

// get approximate distance,
// see bottleneckDistApproxInterval
    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistApprox(PairContainer& A, PairContainer& B,
                         const typename DiagramTraits<PairContainer>::RealType delta,
                         hera::bt::MatchingEdge<typename DiagramTraits<PairContainer>::RealType>& longest_edge,
                         bool compute_longest_edge = true)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(A);
        hera::bt::DiagramPointSet<Real> b(B);
        return hera::bt::bottleneckDistApprox(a, b, delta, longest_edge, compute_longest_edge);
    }

    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistApprox(PairContainer& A, PairContainer& B,
                         const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::MatchingEdge<Real> longest_edge;
        return hera::bottleneckDistApprox(A, B, delta, longest_edge, false);
    }


    // -------------------------------------------------------------------
    // Detailed variants: in addition to the bottleneck distance, return
    // the full matching (`edges`) and every edge tied for the bottleneck
    // cost (`longest_edges`). Ties are reported by exact equality on the
    // per-edge length, which is meaningful e.g. on integer grids under L_inf.
    // -------------------------------------------------------------------
    template<class Real = double>
    struct BottleneckResult
    {
        Real distance { 0 };
        std::vector<hera::bt::MatchingEdge<Real>> edges;
        std::vector<hera::bt::MatchingEdge<Real>> longest_edges;
    };

    template<class PairContainer>
    BottleneckResult<typename DiagramTraits<PairContainer>::RealType>
    bottleneckDetailedApprox(PairContainer& A, PairContainer& B,
                             const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(A);
        hera::bt::DiagramPointSet<Real> b(B);

        BottleneckResult<Real> result;
        hera::bt::MatchingEdge<Real> dummy_longest_edge;
        result.distance = hera::bt::bottleneckDistApprox(
            a, b, delta,
            dummy_longest_edge, /*compute_longest_edge=*/false,
            &result.edges, &result.longest_edges);
        return result;
    }

    template<class PairContainer>
    BottleneckResult<typename DiagramTraits<PairContainer>::RealType>
    bottleneckDetailedExact(PairContainer& A, PairContainer& B,
                            const int decPrecision = 14)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(A);
        hera::bt::DiagramPointSet<Real> b(B);

        BottleneckResult<Real> result;
        hera::bt::MatchingEdge<Real> dummy_longest_edge;
        result.distance = hera::bt::bottleneckDistExact(
            a, b, decPrecision,
            dummy_longest_edge, /*compute_longest_edge=*/false,
            &result.edges, &result.longest_edges);
        return result;
    }


    // ----------------------------------------------------------------------
    // Bucketed bottleneck matching: same shape as WassersteinMatching, plus
    // longest-edge data (the bottleneck distance is, by definition, attained
    // by the longest matched edge).
    // ----------------------------------------------------------------------
    template<class Real>
    struct FiniteLongestEdge {
        Real length { 0 };
        int  idx_a { -1 };   // -1 if this endpoint is a diagonal projection
        int  idx_b { -1 };
        Real a_x { 0 }, a_y { 0 };
        Real b_x { 0 }, b_y { 0 };
    };

    template<class Real>
    struct EssentialLongestEdge {
        Real length { 0 };
        int  idx_a { -1 };
        int  idx_b { -1 };
        Real coord_a { 0 };
        Real coord_b { 0 };
    };

    // BottleneckMatching IS-A WassersteinMatching with extra longest-edge
    // bookkeeping — exposing the inheritance lets nanobind model it directly,
    // so users can write isinstance(bm, DiagramMatching) and get True.
    template<class Real>
    struct BottleneckMatching : WassersteinMatching<Real> {
        std::vector<FiniteLongestEdge<Real>> longest_finite;
        std::array<std::vector<EssentialLongestEdge<Real>>, hera::kNumInfKinds> longest_essential;
    };

    template<class Real>
    inline std::ostream& operator<<(std::ostream& out, const BottleneckMatching<Real>& m)
    {
        int ess_total = 0, ess_long_total = 0;
        for (const auto& v : m.essential) ess_total += static_cast<int>(v.size());
        for (const auto& v : m.longest_essential) ess_long_total += static_cast<int>(v.size());
        out << "BottleneckMatching(finite_to_finite=" << m.finite_to_finite.size()
            << ", a_to_diagonal=" << m.a_to_diagonal.size()
            << ", b_to_diagonal=" << m.b_to_diagonal.size()
            << ", essential=" << ess_total
            << ", longest_finite=" << m.longest_finite.size()
            << ", longest_essential=" << ess_long_total
            << ", distance=" << m.distance << ")";
        return out;
    }

    template<class Real>
    inline std::string to_str_debug(const BottleneckMatching<Real>& m)
    {
        static const char* names[] = {"inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"};
        std::stringstream ss;
        ss << "BottleneckMatching {\n  distance: " << m.distance
           << "\n  finite_to_finite (" << m.finite_to_finite.size() << "):";
        for (auto&& pr : m.finite_to_finite) ss << " (" << pr.first << "," << pr.second << ")";
        ss << "\n  a_to_diagonal:";
        for (auto i : m.a_to_diagonal) ss << " " << i;
        ss << "\n  b_to_diagonal:";
        for (auto i : m.b_to_diagonal) ss << " " << i;
        for (int k = 0; k < hera::kNumInfKinds; ++k) {
            ss << "\n  essential[" << names[k] << "]:";
            for (auto&& pr : m.essential[k])
                ss << " (" << pr.first << "," << pr.second << ")";
        }
        ss << "\n  longest_finite:";
        for (const auto& e : m.longest_finite)
            ss << " (a=" << e.idx_a << ",b=" << e.idx_b << ",len=" << e.length << ")";
        for (int k = 0; k < hera::kNumInfKinds; ++k) {
            ss << "\n  longest_essential[" << names[k] << "]:";
            for (const auto& e : m.longest_essential[k])
                ss << " (a=" << e.idx_a << ",b=" << e.idx_b << ",len=" << e.length << ")";
        }
        ss << "\n}";
        return ss.str();
    }


    namespace bt_detail {
        template<class Real>
        inline int classify_essential(Real x, Real y)
        {
            constexpr Real plus_inf  =  std::numeric_limits<Real>::infinity();
            constexpr Real minus_inf = -std::numeric_limits<Real>::infinity();
            if (std::isfinite(x) && y == plus_inf)  return static_cast<int>(hera::InfKind::InfDeath);
            if (std::isfinite(x) && y == minus_inf) return static_cast<int>(hera::InfKind::NegInfDeath);
            if (x == plus_inf  && std::isfinite(y)) return static_cast<int>(hera::InfKind::InfBirth);
            if (x == minus_inf && std::isfinite(y)) return static_cast<int>(hera::InfKind::NegInfBirth);
            return -1;
        }
    } // namespace bt_detail

    // Build a full bucketed bottleneck matching. Splits each input into
    // finite + four essential families (preserving original indices via
    // user_tag), runs the existing detailed bottleneck on the finite parts,
    // matches each essential family by sorted-rank of the finite coord,
    // and combines longest-edge ties from finite + every essential family.
    //
    // Throws std::invalid_argument if essentials cardinalities differ in any
    // family.
    template<class PairContainer>
    BottleneckMatching<typename DiagramTraits<PairContainer>::RealType>
    bottleneck_matching_detailed(const PairContainer& A,
                                 const PairContainer& B,
                                 typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Traits   = DiagramTraits<PairContainer>;
        using Real     = typename Traits::RealType;
        using DPoint   = hera::DiagramPoint<Real>;
        constexpr Real plus_inf  =  std::numeric_limits<Real>::infinity();
        constexpr Real minus_inf = -std::numeric_limits<Real>::infinity();

        // Split A and B into finite (Hera Diagram) + four essential families.
        std::vector<DPoint> finite_a, finite_b;
        std::array<std::vector<std::pair<Real, int>>, hera::kNumInfKinds> ess_a, ess_b;

        auto split_into = [&](const PairContainer& dgm,
                              std::vector<DPoint>& finite_out,
                              std::array<std::vector<std::pair<Real, int>>, hera::kNumInfKinds>& ess_out)
        {
            for (auto&& p : dgm) {
                Real x = Traits::get_x(p);
                Real y = Traits::get_y(p);
                int  id = Traits::get_id(p);
                if (x == y) continue;  // diagonal noise; drop
                if (std::isfinite(x) && std::isfinite(y)) {
                    // Bottleneck addProjections preserves user_tag through
                    // diagonal projections; set both id and user_tag = original
                    // index so the chosen convention is robust.
                    DPoint dp { x, y, /*id=*/id, /*user_tag=*/id };
                    finite_out.push_back(dp);
                } else {
                    int k = bt_detail::classify_essential(x, y);
                    if (k < 0) continue;  // (inf, inf) etc.: dropped
                    Real coord = (k == static_cast<int>(hera::InfKind::InfDeath) ||
                                  k == static_cast<int>(hera::InfKind::NegInfDeath)) ? x : y;
                    ess_out[k].emplace_back(coord, id);
                }
            }
        };
        split_into(A, finite_a, ess_a);
        split_into(B, finite_b, ess_b);

        BottleneckMatching<Real> result;

        // ----- finite part -----
        Real d_finite = 0;
        if (!finite_a.empty() || !finite_b.empty()) {
            // Identical-diagrams fast path: Hera's bottleneck reduces this
            // to a no-op interval (0, 0) without producing a matching, so
            // we'd otherwise return an empty `edges` list.
            bool finite_identical = false;
            if (finite_a.size() == finite_b.size()) {
                constexpr Real eps = Real(1e-10);
                std::vector<DPoint> a_sorted = finite_a, b_sorted = finite_b;
                auto less = [](const DPoint& p, const DPoint& q) {
                    if (p.getRealX() != q.getRealX()) return p.getRealX() < q.getRealX();
                    return p.getRealY() < q.getRealY();
                };
                std::sort(a_sorted.begin(), a_sorted.end(), less);
                std::sort(b_sorted.begin(), b_sorted.end(), less);
                finite_identical = true;
                for (size_t i = 0; i < a_sorted.size() && finite_identical; ++i) {
                    Real dx = std::abs(a_sorted[i].getRealX() - b_sorted[i].getRealX());
                    Real dy = std::abs(a_sorted[i].getRealY() - b_sorted[i].getRealY());
                    if (dx > eps || dy > eps) finite_identical = false;
                }
            }
            if (finite_identical) {
                // Self-pair every finite point by user_tag (= original index).
                for (const auto& p : finite_a)
                    result.finite_to_finite.emplace_back(p.user_tag, p.user_tag);
                d_finite = Real(0);
                // longest_finite stays empty: every edge has length 0.
                goto finite_done;
            }
            {
            BottleneckResult<Real> br = (delta == Real(0))
                ? bottleneckDetailedExact(finite_a, finite_b)
                : bottleneckDetailedApprox(finite_a, finite_b, delta);
            d_finite = br.distance;

            // Bucket each MatchingEdge into the right slot.
            // Hera's MatchingEdge fields: first (DPoint), second (DPoint).
            // Each DPoint carries user_tag (>=0 for original points,
            //  -1 - other_side_tag for DIAG projections).
            for (const auto& edge : br.edges) {
                int ia = edge.first.is_normal()  ? edge.first.user_tag  : -1;
                int ib = edge.second.is_normal() ? edge.second.user_tag : -1;
                if (ia >= 0 && ib >= 0) {
                    result.finite_to_finite.emplace_back(ia, ib);
                } else if (ia >= 0) {
                    result.a_to_diagonal.push_back(ia);
                } else if (ib >= 0) {
                    result.b_to_diagonal.push_back(ib);
                }
                // diag <-> diag fillers are excluded from `edges` already.
            }

            // longest finite edges
            for (const auto& edge : br.longest_edges) {
                FiniteLongestEdge<Real> fe;
                fe.idx_a = edge.first.is_normal()  ? edge.first.user_tag  : -1;
                fe.idx_b = edge.second.is_normal() ? edge.second.user_tag : -1;
                fe.a_x = edge.first.getRealX();
                fe.a_y = edge.first.getRealY();
                fe.b_x = edge.second.getRealX();
                fe.b_y = edge.second.getRealY();
                if (edge.first.is_diagonal() && edge.second.is_normal()) {
                    fe.length = edge.second.persistence_lp(hera::get_infinity<Real>());
                } else if (edge.first.is_normal() && edge.second.is_diagonal()) {
                    fe.length = edge.first.persistence_lp(hera::get_infinity<Real>());
                } else {
                    fe.length = hera::dist_l_inf(edge.first, edge.second);
                }
                result.longest_finite.push_back(fe);
            }
            }  // end inner block (BottleneckResult br)
        }
        finite_done:

        // ----- essential part -----
        Real d_essential = 0;
        for (int k = 0; k < hera::kNumInfKinds; ++k) {
            auto& ea = ess_a[k];
            auto& eb = ess_b[k];
            if (ea.size() != eb.size()) {
                throw std::invalid_argument(
                    "hera::bottleneck_matching_detailed: essential point "
                    "cardinalities must match between the two diagrams in "
                    "every family.");
            }
            if (ea.empty()) continue;
            std::sort(ea.begin(), ea.end());
            std::sort(eb.begin(), eb.end());

            Real fam_max = 0;
            std::vector<EssentialLongestEdge<Real>> per_pair;
            per_pair.reserve(ea.size());
            for (size_t i = 0; i < ea.size(); ++i) {
                int ia = ea[i].second;
                int ib = eb[i].second;
                Real ca = ea[i].first;
                Real cb = eb[i].first;
                Real len = std::abs(ca - cb);
                result.essential[k].emplace_back(ia, ib);
                per_pair.push_back({len, ia, ib, ca, cb});
                if (len > fam_max) fam_max = len;
            }
            for (const auto& e : per_pair)
                if (e.length == fam_max)
                    result.longest_essential[k].push_back(e);
            if (fam_max > d_essential) d_essential = fam_max;
        }

        result.distance = std::max(d_finite, d_essential);
        // Bottleneck has no q-exponent: cost == distance.
        result.cost = result.distance;
        return result;
    }


} // end namespace hera

#endif
