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

#ifndef HERA_WASSERSTEIN_H
#define HERA_WASSERSTEIN_H

#include <vector>
#include <map>
#include <math.h>
#include <stdexcept>
#include <array>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>

#include "wasserstein/def_debug_ws.h"
#include "wasserstein/basic_defs_ws.h"
#include "wasserstein/auction_params.h"
#include "wasserstein/auction_result.h"
#include "common/diagram_reader.h"
#include "wasserstein/auction_runner_gs.h"
#include "wasserstein/auction_runner_jac.h"


namespace hera
{


namespace ws
{

    // Compare diagrams as multisets, with a small numerical tolerance.
    //
    // Without a tolerance, ULP-level differences (e.g. caused by translating
    // a point cloud by a non-exactly-representable float) make the early
    // exit in `wasserstein_cost_detailed` miss the "essentially identical"
    // case, and the auction is then asked to certify a relative error
    // against a near-zero true cost — an unsatisfiable termination
    // criterion that makes the auction spin forever.
    //
    // The threshold below is the same convention used elsewhere in the
    // pipeline (Oineus's matching fast path uses 1e-10): well above ULP
    // noise, well below any meaningful Wasserstein distance.
    template<class PairContainer>
    inline bool are_equal(const PairContainer& dgm1, const PairContainer& dgm2)
    {
        using Traits = typename hera::DiagramTraits<PairContainer>;
        using PointType = typename Traits::PointType;
        using RealType = typename Traits::RealType;

        constexpr RealType eps = RealType(1e-10);

        auto coord_eq = [eps](RealType a, RealType b) -> bool {
            if (std::isinf(a) || std::isinf(b))
                return a == b;  // both inf with same sign, or neither
            return std::abs(a - b) <= eps;
        };

        std::vector<PointType> v1, v2;
        v1.reserve(dgm1.size());
        v2.reserve(dgm2.size());
        for(auto&& p : dgm1) {
            if (Traits::get_x(p) != Traits::get_y(p))
                v1.push_back(p);
        }
        for(auto&& p : dgm2) {
            if (Traits::get_x(p) != Traits::get_y(p))
                v2.push_back(p);
        }
        if (v1.size() != v2.size())
            return false;

        auto less = [](const PointType& a, const PointType& b) {
            if (Traits::get_x(a) != Traits::get_x(b))
                return Traits::get_x(a) < Traits::get_x(b);
            return Traits::get_y(a) < Traits::get_y(b);
        };
        std::sort(v1.begin(), v1.end(), less);
        std::sort(v2.begin(), v2.end(), less);

        for(size_t i = 0; i < v1.size(); ++i) {
            if (!coord_eq(Traits::get_x(v1[i]), Traits::get_x(v2[i])) ||
                !coord_eq(Traits::get_y(v1[i]), Traits::get_y(v2[i])))
                return false;
        }
        return true;
    }

    // to handle points with one coordinate = infinity
    template<class T, class P, class R>
    inline void get_one_dimensional_cost(std::vector<T>& pts_A, std::vector<T>& pts_B, const P& params, R& result)
    {
        using RealType = typename std::remove_reference<decltype(std::get<0>(pts_A[0]))>::type;

        if (pts_A.size() != pts_B.size()) {
            result.cost = std::numeric_limits<RealType>::infinity();
            return;
        }

        std::sort(pts_A.begin(), pts_A.end());
        std::sort(pts_B.begin(), pts_B.end());

        for(size_t i = 0; i < pts_A.size(); ++i) {
            RealType a = std::get<0>(pts_A[i]);
            RealType b = std::get<0>(pts_B[i]);

            if (params.return_matching and params.match_inf_points) {
                int id_a = std::get<1>(pts_A[i]);
                int id_b = std::get<1>(pts_B[i]);
                result.add_to_matching(id_a, id_b);
            }

            result.cost += std::pow(std::fabs(a - b), params.wasserstein_power);
        }
    }


    template<class RealType>
    struct SplitProblemInput
    {
        std::vector<DiagramPoint<RealType>> A_1;
        std::vector<DiagramPoint<RealType>> B_1;
        std::vector<DiagramPoint<RealType>> A_2;
        std::vector<DiagramPoint<RealType>> B_2;

        std::unordered_map<size_t, size_t> A_1_indices;
        std::unordered_map<size_t, size_t> A_2_indices;
        std::unordered_map<size_t, size_t> B_1_indices;
        std::unordered_map<size_t, size_t> B_2_indices;

        RealType mid_coord { 0.0 };
        RealType strip_width { 0.0 };

        void init_vectors(size_t n)
        {

            A_1_indices.clear();
            A_2_indices.clear();
            B_1_indices.clear();
            B_2_indices.clear();

            A_1.clear();
            A_2.clear();
            B_1.clear();
            B_2.clear();

            A_1.reserve(n / 2);
            B_1.reserve(n / 2);
            A_2.reserve(n / 2);
            B_2.reserve(n / 2);
        }

        void init(const std::vector<DiagramPoint<RealType>>& A,
                  const std::vector<DiagramPoint<RealType>>& B)
        {
            using DiagramPointR = DiagramPoint<RealType>;

            init_vectors(A.size());

            RealType min_sum = std::numeric_limits<RealType>::max();
            RealType max_sum = -std::numeric_limits<RealType>::max();
            for(const auto& p_A : A) {
                RealType s = p_A[0] + p_A[1];
                if (s > max_sum)
                    max_sum = s;
                if (s < min_sum)
                    min_sum = s;
                mid_coord += s;
            }

            mid_coord /= A.size();

            strip_width = 0.25 * (max_sum - min_sum);

            auto first_diag_iter = std::upper_bound(A.begin(), A.end(), 0, [](const int& a, const DiagramPointR& p) { return a < (int)(p.is_diagonal()); });
            size_t num_normal_A_points = std::distance(A.begin(), first_diag_iter);

            // process all normal points in A,
            // projections follow normal points
            for(size_t i = 0; i < A.size(); ++i) {

                assert(i < num_normal_A_points and A.is_normal() or i >= num_normal_A_points and A.is_diagonal());
                assert(i < num_normal_A_points and B.is_diagonal() or i >= num_normal_A_points and B.is_normal());

                RealType s = i < num_normal_A_points ? A[i][0] + A[i][1] : B[i][0] + B[i][1];

                if (s < mid_coord + strip_width) {
                    // add normal point and its projection to the
                    // left half
                    A_1.push_back(A[i]);
                    B_1.push_back(B[i]);
                    A_1_indices[i] = A_1.size() - 1;
                    B_1_indices[i] = B_1.size() - 1;
                }

                if (s > mid_coord - strip_width) {
                    // to the right half
                    A_2.push_back(A[i]);
                    B_2.push_back(B[i]);
                    A_2_indices[i] = A_2.size() - 1;
                    B_2_indices[i] = B_2.size() - 1;
                }

            }
        } // end init

    };

    // CAUTION:
    // this function assumes that all coordinates are finite
    // points at infinity are processed in wasserstein_cost
    template<class RealType>
    inline AuctionResult<RealType> wasserstein_cost_vec_detailed(const std::vector<DiagramPoint<RealType>>& A,
            const std::vector<DiagramPoint<RealType>>& B,
            const AuctionParams<RealType> params)
    {
        if (params.wasserstein_power < 1.0) {
            throw std::runtime_error("Bad q in Wasserstein " + std::to_string(params.wasserstein_power));
        }
        if (params.delta < 0.0) {
            throw std::runtime_error("Bad delta in Wasserstein " + std::to_string(params.delta));
        }
        if (params.initial_epsilon < 0.0) {
            throw std::runtime_error("Bad initial epsilon in Wasserstein" + std::to_string(params.initial_epsilon));
        }
        if (params.epsilon_common_ratio < 0.0) {
            throw std::runtime_error("Bad epsilon factor in Wasserstein " + std::to_string(params.epsilon_common_ratio));
        }

        if (A.empty() and B.empty()) {
            return AuctionResult<RealType>();
        }

        // Tolerant identical-vectors fast path. The auction's relative-error
        // termination criterion can't be satisfied when the true cost is
        // zero (or near zero); calling it with two equal-up-to-ULP point
        // sets makes it spin forever. This happens in practice when the
        // caller (wasserstein_cost_detailed) hands us two diagrams whose
        // finite parts agree but whose essential parts differ.
        if (A.size() == B.size()) {
            constexpr RealType eps = RealType(1e-10);
            using DP = DiagramPoint<RealType>;
            std::vector<DP> a_sorted = A, b_sorted = B;
            auto less = [](const DP& p, const DP& q) {
                if (p.getRealX() != q.getRealX()) return p.getRealX() < q.getRealX();
                return p.getRealY() < q.getRealY();
            };
            std::sort(a_sorted.begin(), a_sorted.end(), less);
            std::sort(b_sorted.begin(), b_sorted.end(), less);
            bool identical = true;
            for (size_t i = 0; i < a_sorted.size() && identical; ++i) {
                RealType dx = std::abs(a_sorted[i].getRealX() - b_sorted[i].getRealX());
                RealType dy = std::abs(a_sorted[i].getRealY() - b_sorted[i].getRealY());
                if (dx > eps || dy > eps) identical = false;
            }
            if (identical) {
                AuctionResult<RealType> r;
                r.cost = 0;
                if (params.return_matching) {
                    // Pair NORMAL_A points with NORMAL_B points and
                    // DIAG_A points with DIAG_B points by id matching.
                    // For the bucketing in wasserstein_matching_detailed
                    // to do the right thing, every NORMAL A point must
                    // be matched to its corresponding NORMAL B counterpart,
                    // and every DIAG slot to its corresponding DIAG slot.
                    // Easy: match by user id (id field). Build maps.
                    std::unordered_map<int, int> a_by_id, b_by_id;
                    for (size_t i = 0; i < A.size(); ++i) a_by_id[A[i].id] = static_cast<int>(i);
                    for (size_t i = 0; i < B.size(); ++i) b_by_id[B[i].id] = static_cast<int>(i);
                    for (auto&& kv : a_by_id) {
                        auto it = b_by_id.find(kv.first);
                        if (it != b_by_id.end()) {
                            r.add_to_matching(kv.first, kv.first);
                        }
                    }
                }
                return r;
            }
        }

        // just use Gauss-Seidel
        AuctionRunnerGS<RealType> auction(A, B, params);

        auction.run_auction();
        return auction.get_result();
    }

    // CAUTION:
    // this function assumes that all coordinates are finite
    // points at infinity are processed in wasserstein_cost
    template<class RealType>
    inline RealType wasserstein_cost_vec(const std::vector<DiagramPoint<RealType>>& A,
                                  const std::vector<DiagramPoint<RealType>>& B,
                                  AuctionParams<RealType>& params)
    {
        if (params.wasserstein_power < 1.0) {
            throw std::runtime_error("Bad q in Wasserstein " + std::to_string(params.wasserstein_power));
        }
        if (params.delta < 0.0) {
            throw std::runtime_error("Bad delta in Wasserstein " + std::to_string(params.delta));
        }
        if (params.initial_epsilon < 0.0) {
            throw std::runtime_error("Bad initial epsilon in Wasserstein" + std::to_string(params.initial_epsilon));
        }
        if (params.epsilon_common_ratio < 0.0) {
            throw std::runtime_error("Bad epsilon factor in Wasserstein " + std::to_string(params.epsilon_common_ratio));
        }

        if (A.empty() and B.empty())
            return 0.0;

        RealType result;

        // just use Gauss-Seidel
        AuctionRunnerGS<RealType> auction(A, B, params);

        auction.run_auction();
        result = auction.get_wasserstein_cost();
        params.final_relative_error = auction.get_relative_error();

        if (params.return_matching) {
            for(size_t bidder_idx = 0; bidder_idx < auction.num_bidders; ++bidder_idx) {
                int bidder_id = auction.get_bidder_id(bidder_idx);
                int item_id = auction.get_bidders_item_id(bidder_idx);
                params.add_to_matching(bidder_id, item_id);
            }
        }

        return result;
    }

} // ws

template<class PairContainer>
inline AuctionResult<typename DiagramTraits<PairContainer>::RealType>
wasserstein_cost_detailed(const PairContainer& A,
        const PairContainer& B,
        const AuctionParams<typename DiagramTraits<PairContainer>::RealType >& params)
{
    using Traits = DiagramTraits<PairContainer>;
    using RealType  = typename Traits::RealType;
    using DgmPoint = hera::DiagramPoint<RealType>;
    using OneDimPoint = std::tuple<RealType, int>;

    constexpr RealType plus_inf = std::numeric_limits<RealType>::infinity();
    constexpr RealType minus_inf = -std::numeric_limits<RealType>::infinity();

    // The auction algorithm is inherently approximate: it terminates when the
    // relative error drops below `delta`. With `delta <= 0` the termination
    // criterion can never be satisfied and the auction will spin forever.
    // Bottleneck distance has a separate exact algorithm (delta == 0 OK there);
    // Wasserstein does not.
    if (params.delta <= RealType(0)) {
        throw std::invalid_argument(
            "hera::wasserstein_cost: delta must be strictly positive "
            "(the auction algorithm has no exact mode for Wasserstein). "
            "Use a small positive value, e.g. delta=0.01.");
    }

    // TODO: return matching here too?
    if (hera::ws::are_equal(A, B)) {
        return AuctionResult<RealType>();
    }

    bool a_empty = true;
    bool b_empty = true;
    RealType total_cost_A = 0;
    RealType total_cost_B = 0;

    std::vector<DgmPoint> dgm_A, dgm_B;
    // points at infinity
    std::vector<OneDimPoint> x_plus_A, x_minus_A, y_plus_A, y_minus_A;
    std::vector<OneDimPoint> x_plus_B, x_minus_B, y_plus_B, y_minus_B;
    // points with both coordinates infinite are treated as equal
    int n_minus_inf_plus_inf_A = 0;
    int n_plus_inf_minus_inf_A = 0;
    int n_minus_inf_plus_inf_B = 0;
    int n_plus_inf_minus_inf_B = 0;
    // loop over A, add projections of A-points to corresponding positions
    // in B-vector
    for(auto&& point_A : A) {
        a_empty = false;
        RealType x = Traits::get_x(point_A);
        RealType y = Traits::get_y(point_A);
        int  id = Traits::get_id(point_A);

        // skip diagonal points, including (inf, inf), (-inf, -inf)
        if (x == y) {
            continue;
        }

        if (x == plus_inf && y == minus_inf) {
            n_plus_inf_minus_inf_A++;
        } else if (x == minus_inf && y == plus_inf) {
            n_minus_inf_plus_inf_A++;
        } else if ( x == plus_inf) {
            y_plus_A.emplace_back(y, Traits::get_id(point_A));
        } else if (x == minus_inf) {
            y_minus_A.emplace_back(y, Traits::get_id(point_A));
        } else if (y == plus_inf) {
            x_plus_A.emplace_back(x, Traits::get_id(point_A));
        } else if (y == minus_inf) {
            x_minus_A.emplace_back(x, Traits::get_id(point_A));
        } else {
            dgm_A.emplace_back(x, y,  DgmPoint::NORMAL, id);
            dgm_B.emplace_back(x, y,  DgmPoint::DIAG, -id - 1);
            total_cost_A += std::pow(dgm_A.back().persistence_lp(params.internal_p), params.wasserstein_power);
        }
    }
    // the same for B
    for(auto&& point_B : B) {
        b_empty = false;
        RealType x = Traits::get_x(point_B);
        RealType y = Traits::get_y(point_B);
        int     id = Traits::get_id(point_B);

        if (x == y) {
            continue;
        }

        if (x == plus_inf && y == minus_inf) {
            n_plus_inf_minus_inf_B++;
        } else if (x == minus_inf && y == plus_inf) {
            n_minus_inf_plus_inf_B++;
        } else if (x == plus_inf) {
            y_plus_B.emplace_back(y, Traits::get_id(point_B));
        } else if (x == minus_inf) {
            y_minus_B.emplace_back(y, Traits::get_id(point_B));
        } else if (y == plus_inf) {
            x_plus_B.emplace_back(x, Traits::get_id(point_B));
        } else if (y == minus_inf) {
            x_minus_B.emplace_back(x, Traits::get_id(point_B));
        } else {
            dgm_A.emplace_back(x, y,  DgmPoint::DIAG, -id - 1);
            dgm_B.emplace_back(x, y,  DgmPoint::NORMAL, id);
            total_cost_B += std::pow(dgm_B.back().persistence_lp(params.internal_p), params.wasserstein_power);
        }
    }

    AuctionResult<RealType> infinity_result;

    if (n_plus_inf_minus_inf_A != n_plus_inf_minus_inf_B || n_minus_inf_plus_inf_A != n_minus_inf_plus_inf_B) {
        infinity_result.cost = plus_inf;
        infinity_result.distance = plus_inf;
        return infinity_result;
    } else {
        ws::get_one_dimensional_cost(x_plus_A, x_plus_B, params, infinity_result);
        ws::get_one_dimensional_cost(x_minus_A, x_minus_B, params, infinity_result);
        ws::get_one_dimensional_cost(y_plus_A, y_plus_B, params, infinity_result);
        ws::get_one_dimensional_cost(y_minus_A, y_minus_B, params, infinity_result);
    }

    if (a_empty) {
        AuctionResult<RealType> b_res;
        b_res.cost = total_cost_B;

        if (params.return_matching) {
            for(size_t b_idx = 0; b_idx < dgm_B.size(); ++b_idx) {
                auto id_a = dgm_A[b_idx].id;
                auto id_b = dgm_B[b_idx].id;
                b_res.add_to_matching(id_a, id_b);
            }
        }

        return add_results(b_res, infinity_result, params.wasserstein_power);
    }

    if (b_empty) {
        AuctionResult<RealType> a_res;
        a_res.cost = total_cost_A;

        if (params.return_matching) {
            for(size_t a_idx = 0; a_idx < dgm_A.size(); ++a_idx) {
                auto id_a = dgm_A[a_idx].id;
                auto id_b = dgm_B[a_idx].id;
                a_res.add_to_matching(id_a, id_b);
            }
        }

        return add_results(a_res, infinity_result, params.wasserstein_power);
    }

    if (infinity_result.cost == plus_inf) {
        return infinity_result;
    } else {
        return add_results(infinity_result, hera::ws::wasserstein_cost_vec_detailed(dgm_A, dgm_B, params), params.wasserstein_power);
    }
}


template<class PairContainer>
inline typename DiagramTraits<PairContainer>::RealType
wasserstein_cost(const PairContainer& A,
                const PairContainer& B,
                const AuctionParams< typename DiagramTraits<PairContainer>::RealType > params)
{
    return wasserstein_cost_detailed(A, B, params).cost;
}

template<class PairContainer>
inline typename DiagramTraits<PairContainer>::RealType
wasserstein_dist(const PairContainer& A,
                 const PairContainer& B,
                 const AuctionParams<typename DiagramTraits<PairContainer>::RealType> params)
{
    using Real = typename DiagramTraits<PairContainer>::RealType;
    return std::pow(hera::wasserstein_cost(A, B, params), Real(1.)/params.wasserstein_power);
}


// ---------------------------------------------------------------------------
// Detailed matching: returns the bucketed matching plus the cost.
//
// The four families of essential (infinite-coordinate) diagram points,
// indexed in this order so callers can use the integer value as an index
// into WassersteinMatching::essential / BottleneckMatching::essential.
// ---------------------------------------------------------------------------
enum class InfKind : int {
    InfDeath     = 0,   // (finite,    +inf)
    NegInfDeath  = 1,   // (finite,    -inf)
    InfBirth     = 2,   // (+inf,    finite)
    NegInfBirth  = 3,   // (-inf,    finite)
};

constexpr int kNumInfKinds = 4;

template<class Real>
struct WassersteinMatching {
    // Indices in every vector are positions in the original input
    // diagrams (preserved through the auction via the user-side `id` field
    // on each point).
    std::vector<std::pair<int, int>> finite_to_finite;
    std::vector<int> a_to_diagonal;
    std::vector<int> b_to_diagonal;
    std::array<std::vector<std::pair<int, int>>, kNumInfKinds> essential;
    Real cost { 0 };
    Real distance { 0 };  // cost ** (1 / wasserstein_power)
};

template<class Real>
inline std::ostream& operator<<(std::ostream& out, const WassersteinMatching<Real>& m)
{
    int ess_total = 0;
    for (const auto& v : m.essential) ess_total += static_cast<int>(v.size());
    out << "WassersteinMatching(finite_to_finite=" << m.finite_to_finite.size()
        << ", a_to_diagonal=" << m.a_to_diagonal.size()
        << ", b_to_diagonal=" << m.b_to_diagonal.size()
        << ", essential=" << ess_total
        << ", distance=" << m.distance << ")";
    return out;
}

template<class Real>
inline std::string to_str_debug(const WassersteinMatching<Real>& m)
{
    static const char* names[] = {"inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"};
    std::stringstream ss;
    ss << "WassersteinMatching {\n"
       << "  cost: " << m.cost << "\n"
       << "  distance: " << m.distance << "\n"
       << "  finite_to_finite (" << m.finite_to_finite.size() << "):";
    for (auto&& pr : m.finite_to_finite) ss << " (" << pr.first << "," << pr.second << ")";
    ss << "\n  a_to_diagonal:";
    for (auto i : m.a_to_diagonal) ss << " " << i;
    ss << "\n  b_to_diagonal:";
    for (auto i : m.b_to_diagonal) ss << " " << i;
    for (int k = 0; k < kNumInfKinds; ++k) {
        ss << "\n  essential[" << names[k] << "]:";
        for (auto&& pr : m.essential[k])
            ss << " (" << pr.first << "," << pr.second << ")";
    }
    ss << "\n}";
    return ss.str();
}


// Build a full bucketed Wasserstein matching by post-processing
// `wasserstein_cost_detailed`'s flat matching:
//   - id < 0 endpoints encode diagonal projections (-orig_id - 1);
//   - both ids >= 0 means either finite-to-finite or essential-to-essential
//     (distinguished by inspecting the original points).
//
// Throws if essentials cardinalities mismatch in any family (Hera reports
// `cost = +inf` in that case; we surface it as an exception so callers
// don't silently get a non-matching result).
template<class PairContainer>
WassersteinMatching<typename DiagramTraits<PairContainer>::RealType>
wasserstein_matching_detailed(const PairContainer& A,
                              const PairContainer& B,
                              const AuctionParams<typename DiagramTraits<PairContainer>::RealType>& params_in)
{
    using Traits   = DiagramTraits<PairContainer>;
    using RealType = typename Traits::RealType;

    AuctionParams<RealType> params = params_in;
    params.return_matching   = true;
    params.match_inf_points  = true;

    auto auction = wasserstein_cost_detailed(A, B, params);

    WassersteinMatching<RealType> result;
    result.cost = auction.cost;
    result.distance = std::pow(result.cost, RealType(1) / params.wasserstein_power);

    constexpr RealType plus_inf  =  std::numeric_limits<RealType>::infinity();
    constexpr RealType minus_inf = -std::numeric_limits<RealType>::infinity();

    if (result.cost == plus_inf) {
        throw std::invalid_argument(
            "hera::wasserstein_matching_detailed: essential point cardinalities "
            "must match between the two diagrams in every family.");
    }

    auto classify = [&](RealType x, RealType y) -> int {
        if (std::isfinite(x) && y == plus_inf)  return static_cast<int>(InfKind::InfDeath);
        if (std::isfinite(x) && y == minus_inf) return static_cast<int>(InfKind::NegInfDeath);
        if (x == plus_inf  && std::isfinite(y)) return static_cast<int>(InfKind::InfBirth);
        if (x == minus_inf && std::isfinite(y)) return static_cast<int>(InfKind::NegInfBirth);
        return -1;
    };

    // Identical-diagrams fast path: wasserstein_cost_detailed returns an
    // empty AuctionResult (cost = 0, no matching). Populate the identity
    // matching ourselves: for each off-diagonal point, pair it with the
    // point at the same id in the other diagram (which our caller is
    // expected to have built via numpy_to_diagram_with_pos_ids).
    if (auction.matching_a_to_b_.empty() && result.cost == RealType(0)) {
        for (auto&& p : A) {
            RealType x = Traits::get_x(p);
            RealType y = Traits::get_y(p);
            int id = Traits::get_id(p);
            if (x == y) continue;  // diagonal noise
            if (std::isfinite(x) && std::isfinite(y)) {
                result.finite_to_finite.emplace_back(id, id);
            } else {
                int k = classify(x, y);
                if (k >= 0)
                    result.essential[k].emplace_back(id, id);
            }
        }
        return result;
    }

    // id -> (x, y) lookup for both diagrams. O(|A| + |B|).
    // PairContainer iteration order isn't guaranteed to match the user-side
    // `id`, so we can't index by position; an unordered_map is robust and
    // more than fast enough at the sizes Hera realistically handles.
    std::unordered_map<int, std::pair<RealType, RealType>> a_pts, b_pts;
    a_pts.reserve(A.size());
    b_pts.reserve(B.size());
    for (auto&& p : A)
        a_pts.emplace(Traits::get_id(p),
                      std::make_pair(Traits::get_x(p), Traits::get_y(p)));
    for (auto&& p : B)
        b_pts.emplace(Traits::get_id(p),
                      std::make_pair(Traits::get_x(p), Traits::get_y(p)));

    for (auto&& kv : auction.matching_a_to_b_) {
        int a_id = kv.first;
        int b_id = kv.second;
        if (a_id < 0 && b_id < 0) {
            // DIAG-to-DIAG filler match between two projections; ignore.
            continue;
        } else if (a_id < 0) {
            // A-side slot is a DIAG projection -> the actual point is on B.
            result.b_to_diagonal.push_back(b_id);
        } else if (b_id < 0) {
            result.a_to_diagonal.push_back(a_id);
        } else {
            auto it = a_pts.find(a_id);
            if (it == a_pts.end()) continue;  // defensive: unknown id
            RealType ax = it->second.first;
            RealType ay = it->second.second;
            if (std::isfinite(ax) && std::isfinite(ay)) {
                result.finite_to_finite.emplace_back(a_id, b_id);
            } else {
                int k = classify(ax, ay);
                if (k >= 0)
                    result.essential[k].emplace_back(a_id, b_id);
            }
        }
    }

    return result;
}

} // end of namespace hera

#endif
