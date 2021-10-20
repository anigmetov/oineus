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

#include "def_debug_ws.h"
#include "basic_defs_ws.h"
#include "diagram_reader.h"
#include "auction_runner_gs.h"


namespace hera
{

template<class PairContainer_, class PointType_ = typename std::remove_reference< decltype(*std::declval<PairContainer_>().begin())>::type >
struct DiagramTraits
{
    using PointType = PointType_;
    using RealType  = typename std::remove_cv<typename std::remove_reference<decltype(std::declval<PointType>()[0])>::type>::type;

    static RealType get_x(const PointType& p)       { return p[0]; }
    static RealType get_y(const PointType& p)       { return p[1]; }
    static id_type  get_id(const PointType& p)      { return p.id; }
};

template<class PairContainer_, class RealType_>
struct DiagramTraits<PairContainer_, std::pair<RealType_, RealType_>>
{
    using RealType  = RealType_;
    using PointType = std::pair<RealType, RealType>;

    static RealType get_x(const PointType& p)       { return p.first; }
    static RealType get_y(const PointType& p)       { return p.second; }
    static id_type  get_id(const PointType&)        { return 0; }
};


namespace ws
{

    // compare as multisets
    template<class PairContainer>
    inline bool are_equal(const PairContainer& dgm1, const PairContainer& dgm2)
    {
        using Traits = typename hera::DiagramTraits<PairContainer>;
        using PointType = typename Traits::PointType;

        std::map<PointType, int> m1, m2;

        for(auto&& pair1 : dgm1) {
            if (Traits::get_x(pair1) != Traits::get_y(pair1))
                m1[pair1]++;
        }

        for(auto&& pair2 : dgm2) {
            if (Traits::get_x(pair2) != Traits::get_y(pair2))
                m2[pair2]++;
        }

        return m1 == m2;
    }

    // to handle points with one coordinate = infinity
    template<class RealType>
    inline RealType get_one_dimensional_cost(std::vector<RealType>& set_A,
            std::vector<RealType>& set_B,
            const RealType wasserstein_power)
    {
        if (set_A.size() != set_B.size()) {
            return std::numeric_limits<RealType>::infinity();
        }
        std::sort(set_A.begin(), set_A.end());
        std::sort(set_B.begin(), set_B.end());
        RealType result = 0.0;
        for(size_t i = 0; i < set_A.size(); ++i) {
            result += std::pow(std::fabs(set_A[i] - set_B[i]), wasserstein_power);
        }
        return result;
    }


    // CAUTION:
    // this function assumes that all coordinates are finite
    // points at infinity are processed in wasserstein_cost
    template<class RealType>
    inline RealType wasserstein_cost_vec(const std::vector<DiagramPoint<RealType>>& A,
                                  const std::vector<DiagramPoint<RealType>>& B,
                                  AuctionParams<RealType>& params,
                                  const std::string& _log_filename_prefix)
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
        AuctionRunnerGS<RealType> auction(A, B, params, _log_filename_prefix);
        auction.run_auction();
        result = auction.get_wasserstein_cost();

        params.final_relative_error = auction.get_relative_error();

        if (params.return_matching) {

            auto& i_to_b = auction.get_items_to_bidders();

            params.matching_a_to_b_.clear();
            params.matching_b_to_a_.clear();

            for(size_t i = 0; i < static_cast<id_type>(i_to_b.size()); ++i) {
                size_t b = i_to_b.at(i);
                id_type item_id = auction.get_item_id(i);
                id_type bidder_id = auction.get_bidder_id(b);
                params.matching_a_to_b_[bidder_id] = item_id;
                params.matching_b_to_a_[item_id] = bidder_id;
            }
        }

        return result;
    }

} // ws



template<class PairContainer, class Traits = DiagramTraits<PairContainer>>
inline typename DiagramTraits<PairContainer>::RealType
wasserstein_cost(const PairContainer& A,
                const PairContainer& B,
                AuctionParams< typename DiagramTraits<PairContainer>::RealType >& params,
                const std::string& _log_filename_prefix = "")
{
    using RealType  = typename Traits::RealType;

    constexpr RealType plus_inf = std::numeric_limits<RealType>::infinity();
    constexpr RealType minus_inf = -std::numeric_limits<RealType>::infinity();

    if (hera::ws::are_equal(A, B)) {
        return 0.0;
    }

    bool a_empty = true;
    bool b_empty = true;
    RealType total_cost_A = 0.0;
    RealType total_cost_B = 0.0;

    using DgmPoint = hera::ws::DiagramPoint<RealType>;

    std::vector<DgmPoint> dgm_A, dgm_B;
    // coordinates of points at infinity
    std::vector<RealType> x_plus_A, x_minus_A, y_plus_A, y_minus_A;
    std::vector<RealType> x_plus_B, x_minus_B, y_plus_B, y_minus_B;
    // points with both coordinates infinite are treated as equal
    int n_minus_inf_plus_inf_A = 0;
    int n_plus_inf_minus_inf_A = 0;
    int n_minus_inf_plus_inf_B = 0;
    int n_plus_inf_minus_inf_B = 0;
    // loop over A, add projections of A-points to corresponding positions
    // in B-vector
    for(auto&& point_a : A) {
        a_empty = false;
        RealType x = Traits::get_x(point_a);
        RealType y = Traits::get_y(point_a);
        id_type  id = Traits::get_id(point_a);

        // skip diagonal points, including (inf, inf), (-inf, -inf)
        if (x == y) {
            continue;
        }

        if (x == plus_inf && y == minus_inf) {
            n_plus_inf_minus_inf_A++;
        } else if (x == minus_inf && y == plus_inf) {
            n_minus_inf_plus_inf_A++;
        } else if ( x == plus_inf) {
            y_plus_A.push_back(y);
        } else if (x == minus_inf) {
            y_minus_A.push_back(y);
        } else if (y == plus_inf) {
            x_plus_A.push_back(x);
        } else if (y == minus_inf) {
            x_minus_A.push_back(x);
        } else {
            dgm_A.emplace_back(x, y,  DgmPoint::NORMAL, id);
            dgm_B.emplace_back(x, y,  DgmPoint::DIAG, -id - 1);     // use negative id for diagonal projection
            total_cost_A += std::pow(dgm_A.back().persistence_lp(params.internal_p), params.wasserstein_power);
        }
    }
    // the same for B
    for(auto&& point_b : B) {
        b_empty = false;
        RealType x = Traits::get_x(point_b);
        RealType y = Traits::get_y(point_b);
        id_type id = Traits::get_id(point_b);

        if (x == y) {
            continue;
        }

        if (x == plus_inf && y == minus_inf) {
            n_plus_inf_minus_inf_B++;
        } else if (x == minus_inf && y == plus_inf) {
            n_minus_inf_plus_inf_B++;
        } else if (x == plus_inf) {
            y_plus_B.push_back(y);
        } else if (x == minus_inf) {
            y_minus_B.push_back(y);
        } else if (y == plus_inf) {
            x_plus_B.push_back(x);
        } else if (y == minus_inf) {
            x_minus_B.push_back(x);
        } else {
            dgm_A.emplace_back(x, y,  DgmPoint::DIAG, -id - 1);
            dgm_B.emplace_back(x, y,  DgmPoint::NORMAL, id);
            total_cost_B += std::pow(dgm_B.back().persistence_lp(params.internal_p), params.wasserstein_power);
        }
    }

    RealType infinity_cost = 0;

    if (n_plus_inf_minus_inf_A != n_plus_inf_minus_inf_B || n_minus_inf_plus_inf_A != n_minus_inf_plus_inf_B)
        infinity_cost = plus_inf;
    else {
        infinity_cost += ws::get_one_dimensional_cost(x_plus_A, x_plus_B, params.wasserstein_power);
        infinity_cost += ws::get_one_dimensional_cost(x_minus_A, x_minus_B, params.wasserstein_power);
        infinity_cost += ws::get_one_dimensional_cost(y_plus_A, y_plus_B, params.wasserstein_power);
        infinity_cost += ws::get_one_dimensional_cost(y_minus_A, y_minus_B, params.wasserstein_power);
    }

    if (a_empty)
        return total_cost_B + infinity_cost;

    if (b_empty)
        return total_cost_A + infinity_cost;

    if (infinity_cost == plus_inf) {
        return infinity_cost;
    } else {
        return infinity_cost + wasserstein_cost_vec(dgm_A, dgm_B, params, _log_filename_prefix);
    }

}

template<class PairContainer>
inline typename DiagramTraits<PairContainer>::RealType
wasserstein_dist(const PairContainer& A,
                 const PairContainer& B,
                 AuctionParams<typename DiagramTraits<PairContainer>::RealType>& params,
                 const std::string& _log_filename_prefix = "")
{
    using Real = typename DiagramTraits<PairContainer>::RealType;
    return std::pow(hera::wasserstein_cost(A, B, params, _log_filename_prefix), Real(1.)/params.wasserstein_power);
}

} // end of namespace hera

#endif
