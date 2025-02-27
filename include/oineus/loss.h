#ifndef OINEUS_LOSS_H
#define OINEUS_LOSS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <cmath>

#include "diagram.h"
#include "timer.h"
#include "filtration.h"
#include "decomposition.h"

// suppress pragma message from boost
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#include "hera/wasserstein.h"

#pragma once

namespace oineus {

    template<class Cont>
    std::string container_to_string(const Cont& v)
    {
        std::stringstream ss;
        ss << "[";
        for(auto x_iter = v.begin(); x_iter != v.end();) {
            ss << *x_iter;
            x_iter = std::next(x_iter);
            if (x_iter != v.end())
                ss << ", ";
        }
        ss << "]";
        return ss.str();
    }

    enum class DenoiseStrategy {
        BirthBirth,
        DeathDeath,
        Midway
    };

    inline std::ostream& operator<<(std::ostream& out, const DenoiseStrategy& s)
    {
        if (s == DenoiseStrategy::BirthBirth)
            out << "bb";
        else if (s == DenoiseStrategy::DeathDeath)
            out << "dd";
        else if (s == DenoiseStrategy::Midway)
            out << "mid";
        return out;
    }

    inline std::string denoise_strategy_to_string(const DenoiseStrategy& s)
    {
        std::stringstream ss;
        ss << s;
        return ss.str();
    }

    enum class ConflictStrategy {
        Max,
        Avg,
        Sum,
        FixCritAvg
    };

    inline std::ostream& operator<<(std::ostream& out, const ConflictStrategy& s)
    {
        if (s == ConflictStrategy::Max)
            out << "max";
        else if (s == ConflictStrategy::Avg)
            out << "avg";
        else if (s == ConflictStrategy::Sum)
            out << "sum";
        else if (s == ConflictStrategy::FixCritAvg)
            out << "fca";
        //out << "prescribed on critical, average on others";
        return out;
    }

    inline std::string conflict_strategy_to_string(const ConflictStrategy& s)
    {
        std::stringstream ss;
        ss << s;
        return ss.str();
    }

    template<class Real>
    typename Diagrams<Real>::Point denoise_point(Real birth, Real death, DenoiseStrategy s)
    {
        Real target_birth {birth};
        Real target_death {death};

        switch(s) {
        case DenoiseStrategy::BirthBirth : target_death = birth;
            break;
        case DenoiseStrategy::DeathDeath: target_birth = death;
            break;
        case DenoiseStrategy::Midway : target_birth = target_death = (birth + death) / 2;
            break;
        }

        return {target_birth, target_death};
    }

    template<class Real>
    typename Diagrams<Real>::Point enhance_point(Real birth, Real death, Real min_birth = -std::numeric_limits<Real>::infinity(), Real max_death = std::numeric_limits<Real>::infinity())
    {
        Real d = (death - birth) / 2;
        if (death > birth) {
            Real new_birth = std::max(birth - d, min_birth);
            Real new_death = std::min(death + d, max_death);
            return {new_birth, new_death};
        } else // for upper-star signs are reversed
            return {birth + d, death - d};
    }

    template<class Real>
    using DiagramToValues = std::unordered_map<DgmPoint<Real>, DgmPoint<Real>>;

    template<class ValueLocation, class Real>
    using TargetMatching = std::vector<std::pair<ValueLocation, Real>>;

    using Permutation = std::map<size_t, size_t>;

    //template<class Cell, class Real>
    //void match_diagonal_points(const typename oineus::Filtration<Cell, Real>& current_fil,
    //        const typename Diagrams<Real>::Dgm& current_dgm,
    //        typename Diagrams<Real>::Dgm& template_dgm,
    //        typename hera::AuctionResult<Real>& hera_res,
    //        DiagramToValues<Real>& result)
    //{
    //    using Int = typename Cell::Int;
    //    using Diagram = typename Diagrams<Real>::Dgm;
    //    using DgmPt = typename Diagrams<Real>::DgmPoint;
    //    // diagonal point with id
    //    using DiagP = std::tuple<Real, size_t>;
    //    using VecDiagP = std::vector<DiagP>;
    //    VecDiagP current_diagonal_points;

    //    for(size_t current_dgm_id = 0; current_dgm_id < current_dgm.size(); ++current_dgm_id) {
    //        const DgmPt& current_point = current_dgm[current_dgm.id];

    //        if (current_point.is_diagonal() and not current_point.is_inf())
    //            current_diagonal_points.emplace_back(current_point.birth, current_dgm_id);
    //    }

    //    // get unmatched template points
    //    Diagram unmatched_template_points;
    //    for(auto curr_template: hera_res.matching_b_to_a_) {
    //        auto current_id = curr_template.first;
    //        auto template_id = curr_template.second;

    //        if (current_id < 0 and template_id >= 0)
    //            unmatched_template_points.emplace_back(template_dgm[template_id]);
    //    }

    //    if (unmatched_template_points.size() < current_diagonal_points.size())
    //        throw std::runtime_error("Not implemented");

    //    if (unmatched_template_points.size() > current_diagonal_points.size()) {
    //        // keep most persistent points, operator < for points sorts by persistence first
    //        std::sort(unmatched_template_points.begin(), unmatched_template_points.end(), std::greater<DgmPoint<Real>>());
    //        unmatched_template_points.resize(current_diagonal_points.size());
    //    }

    //    std::vector<std::tuple<Real, size_t>> diag_unmatched_template;

    //    for(auto p: unmatched_template_points)
    //        diag_unmatched_template.emplace_back((p.birth + p.death) / 2, p.id);

    //    throw std::runtime_error("todo");
    //    //hera_params.clear_matching();

    //    //hera::ws::get_one_dimensional_cost(diag_unmatched_template, current_diagonal_points, hera_params);

    //    //for(auto curr_template: hera_res.matching_b_to_a_) {
    //    //    auto current_id = curr_template.first;
    //    //    auto template_id = curr_template.second;

    //    //    if (current_id < 0 or template_id < 0)
    //    //        throw std::runtime_error("negative ids in one-dimensional call");

    //    //    result[current_index_dgm.at(current_id)] = template_dgm.at(template_id);
    //    //}
    //}

//    template<class Cell>
//    DiagramToValues<Real> get_barycenter_target(const Filtration<Cell, Real>& fil, VRUDecomposition<typename Cell::Int>& rv, dim_type d, bool is_vr)
//    {
//        using Real = Real;
//        DiagramToValues<Real> result;
//
//        auto index_dgm = rv.index_diagram(fil, false, false).get_diagram_in_dimension(d);
//        auto dgm = rv.diagram(fil, false).get_diagram_in_dimension(d);
//
//        if (dgm.size()) {
//            Real avg_birth = (is_vr ? 0 : std::accumulate(dgm.begin(), dgm.end(), static_cast<Real>(0), [&](auto x, auto p) { return x + p.birth; })) / dgm.size();
//            Real avg_death = std::accumulate(dgm.begin(), dgm.end(), static_cast<Real>(0), [&](auto x, auto p) { return x + p.death; }) / dgm.size();
//
//            for(auto p: index_dgm)
//                if (fil.value_by_sorted_id(p.birth) != fil.value_by_sorted_id(p.death))
//                    result[p] = {avg_birth, avg_death};
//        }
//
//        return result;
//    }
//
//    template<class Cell>
//    DiagramToValues<Real> get_bruelle_target(const Filtration<Cell, Real>& current_fil,
//            VRUDecomposition<typename Cell::Int>& rv,
//            int p,
//            int q,
//            int i_0,
//            dim_type d,
//            bool minimize,
//            bool is_vr,
//            Real min_birth = -std::numeric_limits<Real>::infinity(),
//            Real max_death = std::numeric_limits<Real>::infinity())
//    {
//        if (q == 0 and p == 2)
//            return get_bruelle_target_2_0(current_fil, rv, i_0, d, minimize, min_birth, max_death, is_vr);
//        else
//            throw std::runtime_error("Not implemented");
//    }

//    template<class Cell>
//    DiagramToValues<Real> get_bruelle_target_2_0(const Filtration<Cell, Real>& current_fil,
//            VRUDecomposition<typename Cell::Int>& rv,
//            int n_keep,
//            dim_type d,
//            bool minimize,
//            Real min_birth,
//            Real max_death,
//            bool is_vr)
//    {
//        using Real = Real;
//        DiagramToValues<Real> result;
//
//        Real epsilon = get_nth_persistence(current_fil, rv, d, n_keep);
//
//        // false flags: no infinite points, no points with zero persistence
//        auto index_dgm = rv.index_diagram(current_fil, false, false).get_diagram_in_dimension(d);
//
//        for(auto p: index_dgm) {
//            Real birth_val = current_fil.value_by_sorted_id(p.birth);
//            Real death_val = current_fil.value_by_sorted_id(p.death);
//            if (abs(death_val - birth_val) <= epsilon) {
//                result[p] = minimize ? denoise_point(birth_val, death_val, DenoiseStrategy::Midway) : enhance_point(birth_val, death_val, min_birth, max_death);
//            }
//        }
//
//        return result;
//    }
//

    template<class Real>
    Real clamp(Real a, Real min, Real max)
    {
        if (min > max)
            throw std::runtime_error("bad clamp call");

        if (a > max)
            return max;
        else if (a < min)
            return min;
        else
            return a;
    }


    // return the n-th (finite) persistence value in dimension d
    // points at infinity are ignored
    template<class Cell, class Real>
    Real get_nth_persistence(const Filtration<Cell, Real>& fil, const VRUDecomposition<typename Cell::Int>& rv_matrix, dim_type d, int n)
    {
        if (n < 1) {
            throw std::runtime_error("get_nth_persistence: n must be at least 1");
        }

        using DPoint = typename Diagrams<Real>::Point;

        auto diagram = rv_matrix.diagram(fil, false).get_diagram_in_dimension(d);

        if (diagram.size() >= static_cast<size_t>(n)) {
            std::nth_element(diagram.begin(), diagram.begin() + n - 1, diagram.end(), std::greater<DPoint>());
            return diagram[n - 1].persistence();
        } else {
            return 0;
        }
    }

    // points (b, d) with persistence | b - d| <= eps should go to (b, b)
    template<class Cell, class Real>
    DiagramToValues<Real> get_denoise_target(dim_type d, const Filtration<Cell, Real>& fil, const VRUDecomposition<typename Cell::Int>& rv_matrix, Real eps, DenoiseStrategy strategy)
    {
        DiagramToValues<Real> result;
        auto diagram = rv_matrix.diagram(fil, false)[d];

        for(auto p: diagram) {
            if (p.persistence() <= eps)
                result[p] = denoise_point(p.birth, p.death, strategy);
        }

        return result;
    }

    template<class Real>
    TargetMatching<size_t, Real> get_prescribed_simplex_values_diagram_loss(const DiagramToValues<Real>& diagram_to_values, bool death_only)
    {
        TargetMatching<size_t, Real> result;
        result.reserve(2 * diagram_to_values.size());

        for(auto[dgm_point, target_point]: diagram_to_values) {
            result.emplace_back(dgm_point.death, target_point.death);
            // for Vietoris--Rips filtration changing birth value of vertex is impossible
            if (not death_only) {
                result.emplace_back(dgm_point.birth, target_point.birth);
            }
        }

        return result;
    }

    template<class Cell, class Real>
    std::vector<typename Cell::Int> increase_birth_x([[maybe_unused]] dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Cell, Real>& fil, const oineus::VRUDecomposition<typename Cell::Int>& decmp, Real target_birth)
    {
        if (not decmp.dualize())
            throw std::runtime_error("increase_birth_x: expected cohomology");

        if (not fil.cmp(fil.cells()[positive_simplex_idx].get_value(), target_birth))
            throw std::runtime_error("target_birth cannot preceed current value");

        using Int = typename Cell::Int;

        std::vector<Int> result;

        auto& v_col = decmp.v_data.at(fil.index_in_matrix(positive_simplex_idx, decmp.dualize()));

        for(auto index_in_matrix = v_col.rbegin(); index_in_matrix != v_col.rend(); ++index_in_matrix) {
            auto fil_idx = fil.index_in_filtration(*index_in_matrix, decmp.dualize());
            const auto& sigma = fil.cells().at(fil_idx);

            if (fil.cmp(target_birth, sigma.get_value()))
                break;

            result.push_back(fil_idx);
        }

        if (result.empty())
            throw std::runtime_error("increase_birth_x: empty");

        return result;
    }

    template<class Cell, class Real>
    std::vector<typename Cell::Int> decrease_birth_x([[maybe_unused]] dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Cell, Real>& fil, const oineus::VRUDecomposition<typename Cell::Int>& decmp, Real target_birth)
    {
        if (not decmp.dualize())
            throw std::runtime_error("expected cohomology");

        if (not fil.cmp(target_birth, fil.cells()[positive_simplex_idx].get_value()))
            throw std::runtime_error("target_birth cannot preceed current value");

        using Int = typename Cell::Int;

        std::vector<Int> result;

        for(auto index_in_matrix: decmp.u_data_t.at(fil.index_in_matrix(positive_simplex_idx, decmp.dualize()))) {
            auto fil_idx = fil.index_in_filtration(index_in_matrix, decmp.dualize());
            const auto& sigma = fil.cells()[fil_idx];

            if (fil.cmp(sigma.get_value(), target_birth)) {
                break;
            }

            result.push_back(fil_idx);
        }

        if (result.empty())
            throw std::runtime_error("decrease_birth_x: empty");

        return result;
    }

    template<class Cell, class Real>
    std::vector<typename Cell::Int> increase_death_x([[maybe_unused]] dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Cell, Real>& fil, const oineus::VRUDecomposition<typename Cell::Int>& decmp, Real target_death)
    {
        if (decmp.dualize())
            throw std::runtime_error("increase_death_x: expected homology, got cohomology");

        using Int = typename Cell::Int;

        std::vector<Int> result;

        const auto& u_rows = decmp.u_data_t;
        const auto& r_cols = decmp.r_data;
        const auto& simplices = fil.cells();

        Int sigma = low(r_cols[negative_simplex_idx]);

        if (not(sigma >= 0 and sigma < r_cols.size()))
            throw std::runtime_error("expected negative simplex");

        for(auto tau_idx: u_rows.at(negative_simplex_idx)) {
            const auto& tau = simplices.at(tau_idx);
            assert(tau.dim() == d);
            if (fil.cmp(target_death, tau.get_value())) {
                break;
            }

            if (low(decmp.r_data[tau_idx]) <= sigma)
                result.push_back(tau_idx);
        }

        if (result.empty())
            throw std::runtime_error("increase_death_x: empty");

        return result;
    }

    template<class Cell, class Real>
    std::vector<typename Cell::Int> decrease_death_x([[maybe_unused]] dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Cell, Real>& fil, const oineus::VRUDecomposition<typename Cell::Int>& decmp, Real target_death)
    {
        if (decmp.dualize())
            throw std::runtime_error("decrease_death_x: expected homology, got cohomology");

        using Int = typename Cell::Int;

        std::vector<Int> result;

        auto& r_cols = decmp.r_data;
        Int sigma = low(r_cols[negative_simplex_idx]);

        assert(sigma >= 0 and sigma < r_cols.size());


        auto& v_col = decmp.v_data[negative_simplex_idx];

        for(auto tau_idx_it = v_col.rbegin(); tau_idx_it != v_col.rend(); ++tau_idx_it) {
            auto tau_idx = *tau_idx_it;
            const auto& tau = fil.cells()[tau_idx];
            assert(tau.dim() == d + 1);

            if (fil.cmp(tau.get_value(), target_death))
                break;

            // explicit check for is_zero is not necessary for signed Int, low returns -1 for empty columns
            if (low(decmp.r_data[tau_idx]) < sigma or is_zero(decmp.r_data[tau_idx]))
                continue;

            result.push_back(tau_idx);
        }

        assert(result.size());

        assert(result[0] == negative_simplex_idx);

        return result;
    }

    template<class Cell, class Real>
    std::vector<typename Cell::Int> change_death_x(dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Cell, Real>& fil, const oineus::VRUDecomposition<typename Cell::Int>& decmp, Real target_death)
    {
        auto current_death = fil.cells()[negative_simplex_idx].get_value();
        if (fil.cmp(target_death, current_death))
            return decrease_death_x(d, negative_simplex_idx, fil, decmp, target_death);
        else if (fil.cmp(current_death, target_death))
            return increase_death_x(d, negative_simplex_idx, fil, decmp, target_death);
        else // current_death = target_death, no change required
            return {};
    }

    template<class Cell, class Real>
    std::vector<typename Cell::Int> change_birth_x(dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Cell, Real>& fil, const oineus::VRUDecomposition<typename Cell::Int>& decmp_coh, Real target_birth)
    {
        auto current_birth = fil.cells()[positive_simplex_idx].get_value();
        if (fil.cmp(target_birth, current_birth))
            return decrease_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
        else if (fil.cmp(current_birth, target_birth))
            return increase_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
        else
            return {};
    }

    template<class Cell, class Real>
    TargetMatching<size_t, Real> get_prescribed_simplex_values_set_x(dim_type d,
            const DiagramToValues<Real>& diagram_to_values,
            const oineus::Filtration<Cell, Real>& fil,
            const oineus::VRUDecomposition<typename Cell::Int>& decmp_hom,
            const oineus::VRUDecomposition<typename Cell::Int>& decmp_coh,
            ConflictStrategy conflict_strategy,
            bool death_only)
    {
        if (decmp_hom.dualize())
            throw std::runtime_error("decmp_hom: this parameter must be homology");

        if (not decmp_coh.dualize())
            throw std::runtime_error("this parameter must be cohomology");

        std::unordered_map<size_t, std::vector<Real>> result;

        std::unordered_map<size_t, Real> critical_prescribed;

        for(auto&& dgm_to_target: diagram_to_values) {
            size_t death_idx = dgm_to_target.first.death;
            Real target_death = dgm_to_target.second.death;

            if (target_death != fil.value_by_sorted_id(death_idx)) {
                for(auto d_idx: change_death_x(d, death_idx, fil, decmp_hom, target_death)) {
                    result[d_idx].push_back(target_death);
                }

                assert(critical_prescribed.count(death_idx) == 0);
                critical_prescribed[death_idx] = target_death;
            }

            if (death_only)
                continue;

            size_t birth_idx = dgm_to_target.first.birth;
            Real target_birth = dgm_to_target.second.birth;

            if (target_birth != fil.value_by_sorted_id(birth_idx)) {

                for(auto b_idx: change_birth_x(d, birth_idx, fil, decmp_coh, target_birth)) {
                    result[b_idx].push_back(target_birth);
                }

                assert(critical_prescribed.count(birth_idx) == 0);
                critical_prescribed[birth_idx] = target_birth;
            }
        }

        TargetMatching<size_t, Real> final_result;

        if (conflict_strategy == ConflictStrategy::Max) {
            for(auto&&[simplex_idx, values]: result) {
                Real current_value = fil.value_by_sorted_id(simplex_idx);
                // compare by displacement from current value
                Real target_value = *std::max_element(values.begin(), values.end(), [current_value](Real a, Real b) { return abs(a - current_value) < abs(b - current_value); });
                final_result.emplace_back(simplex_idx, target_value);
            }
        } else if (conflict_strategy == ConflictStrategy::Avg) {
            for(auto&&[simplex_idx, values]: result) {
                Real target_value = std::accumulate(values.begin(), values.end(), static_cast<Real>(0)) / values.size();
                final_result.emplace_back(simplex_idx, target_value);
            }
        } else if (conflict_strategy == ConflictStrategy::Sum) {
            // return all prescribed values, gradient of loss will be summed
            for(auto&&[simplex_idx, values]: result) {
                for(auto value: values) {
                    final_result.emplace_back(simplex_idx, value);
                }
            }
        } else if (conflict_strategy == ConflictStrategy::FixCritAvg) {
            // send critical cells according to the matching loss
            // average on others
            for(auto&&[simplex_idx, values]: result) {
                // where matching loss tells critical cells to go
                // is contained in critical_prescribed map
                auto critical_iter = critical_prescribed.find(simplex_idx);

                Real target_value;
                if (critical_iter == critical_prescribed.end())
                    target_value = std::accumulate(values.begin(), values.end(), static_cast<Real>(0)) / values.size();
                else
                    target_value = critical_iter->second;

                final_result.emplace_back(simplex_idx, target_value);
            }
        }

        return final_result;
    }

    template<class Cell, class Real>
    Permutation targets_to_permutation_naive(const TargetMatching<size_t, Real>& simplex_to_values, const Filtration<Cell,Real>& fil)
    {
        Permutation result;

        using DimValueIndex = std::tuple<dim_type, Real, size_t>;

        std::vector<DimValueIndex> new_vals;

        size_t idx = 0;

        const bool negate = fil.negate();

        // copy current values from filtration
        for(const auto& sigma: fil.cells()) {
            if (negate)
                new_vals.emplace_back(sigma.dim(), -sigma.get_value(), idx++);
            else
                new_vals.emplace_back(sigma.dim(), sigma.get_value(), idx++);
        }

        // update according to diagram_to_values

        for(auto[simplex_idx, value]: simplex_to_values) {
            auto dim = fil.cells()[simplex_idx].dim();

            if (negate)
                value = -value;

            assert(std::get<2>(new_vals[simplex_idx]) == simplex_idx);

            new_vals[simplex_idx] = {dim, value, simplex_idx};
        }

        // tuples are automatically compared lexicographically, dimension -> value -> old index
        std::sort(new_vals.begin(), new_vals.end());

        // record indices that are not mapped to itself
        for(size_t new_index = 0; new_index < new_vals.size(); ++new_index) {
            auto old_index = std::get<2>(new_vals[new_index]);
            if (new_index != old_index) {
                assert(result.count(old_index) == 0);
                result[old_index] = new_index;
            }
        }

        return result;
    }

    template<class Cell, class Real>
    Permutation targets_to_permutation_dtv(const DiagramToValues<Real>& diagram_to_values, const Filtration<Cell, Real>& fil)
    {
        auto tm = get_prescribed_simplex_values_diagram_loss(diagram_to_values, false);
        return targets_to_permutation_naive(tm, fil);
    }

    template<class Cell, class Real>
    Permutation targets_to_permutation(const TargetMatching<size_t, Real>& simplex_to_values, const Filtration<Cell, Real>& fil)
    {
        // TODO: find a better solution
        return targets_to_permutation_naive(simplex_to_values, fil);
    }

} // namespace oineus


#endif //OINEUS_LOSS_H
