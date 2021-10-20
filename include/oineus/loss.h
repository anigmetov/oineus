#ifndef OINEUS_LOSS_H
#define OINEUS_LOSS_H

#include <vector>
#include <cassert>
#include <unordered_map>
#include <cmath>

#include "diagram.h"
#include "timer.h"
#include "filtration.h"
#include "matrix.h"
#include "hera/wasserstein.h"

#pragma once

namespace oineus {

template<class Real>
using DiagramToValues = std::unordered_map<DgmPoint<size_t>, DgmPoint<Real>>;

template<class ValueLocation, class Real>
using TargetMatching = std::unordered_map<ValueLocation, Real>;

// evaluate linear function f such that on f(x_1,2) = y_1,2 on xs
template<class Real>
std::vector<Real> lin_interp(std::vector<Real> xs, Real x_1, Real x_2, Real y_1, Real y_2)
{
    Real k = (y_2 - y_1) / (x_2 - x_1);
    Real b = y_1 - k * x_1;

    std::vector<Real> ys;
    ys.reserve(xs.size());

    for(auto x: xs)
        ys.push_back(k * x + b);

    return ys;
}


template<class Int, class Real, class L>
DiagramToValues<Real> get_target_from_matching(typename Diagrams<Real>::Dgm& template_dgm,
                                               const Filtration<Int, Real, L>& current_fil,
                                               SparseMatrix<Int>& rv,
                                               dim_type d,
                                               Real wasserstein_q=1)
{
    if (not rv.is_reduced) {
        std::cerr << "Warning: get_current_from_matching expects reduced matrix; reducing with default reduction parameters" << std::endl;
        Params params;
        rv.reduce_parallel(params);
    }

    for(hera::id_type i = 0; i < template_dgm.size(); ++i) {
        template_dgm[i].id = i;

        if (template_dgm[i].is_inf())
            throw std::runtime_error("infinite point in template diagram");
    }

    using Diagram = typename Diagrams<Real>::Dgm;

    DiagramToValues<Real> result;

    hera::AuctionParams<Real> hera_params;
    hera_params.return_matching = true;
    hera_params.wasserstein_power = wasserstein_q;

    auto current_index_dgm = rv.index_diagram_finite(current_fil).get_diagram_in_dimension(d);

    Diagram current_dgm;
    current_dgm.reserve(current_index_dgm.size());

    for(hera::id_type current_dgm_id = 0; current_dgm_id < current_index_dgm.size(); ++current_dgm_id) {

        auto birth_idx = current_index_dgm[current_dgm_id].birth;
        auto death_idx = current_index_dgm[current_dgm_id].death;

        auto birth_val = current_fil.value_by_sorted_id(birth_idx);
        auto death_val = current_fil.value_by_sorted_id(death_idx);

        current_dgm.emplace_back(birth_val, death_val, current_dgm_id);
    }

    // template_dgm: bidders, a
    // current_dgm: items, b
    hera::wasserstein_cost<Diagram>(template_dgm, current_dgm, hera_params);

    for(auto curr_template : hera_params.matching_b_to_a_) {
        auto current_id = curr_template.first;
        auto template_id = curr_template.second;

        if (current_id < 0)
            continue;

        if (template_id >= 0)
            // matched to off-diagonal point of template diagram
            result[current_index_dgm.at(current_id)] = template_dgm.at(template_id);
        else {
            // point must disappear, move to (birth, birth)
            auto target_point = current_dgm.at(current_id);
            target_point.death = target_point.birth;
            result[current_index_dgm.at(current_id)] = target_point;
        }
    }

    return result;
}



// points (b, d) with persistence | b - d| <= eps should go to (b, b)
template<class Int, class Real, class L>
DiagramToValues<Real> get_denoise_target(dim_type d, const Filtration<Int, Real, L>& fil, const SparseMatrix<Int>& rv_matrix, Real eps)
{
    DiagramToValues<Real> result;
    auto index_diagram = rv_matrix.template index_diagram_finite<Real, L>(fil)[d];

    for(auto p : index_diagram) {
        Real birth_value = fil.simplices()[p.birth].value();
        Real death_value = fil.simplices()[p.death].value();
        Real pers =abs(death_value - birth_value);
        if (pers > Real(0) and pers <= eps) {
            Real target_birth = birth_value;
            Real target_death = birth_value;
            result[p] = {target_birth, target_death};
        }
    }

    return result;
}

// given target points (diagram_to_values), compute values on intermediate simplices from R, V columns
template<class Int, class Real, class L>
TargetMatching<L, Real> get_target_values_diagram_loss(dim_type d, const DiagramToValues<Real>& diagram_to_values, const oineus::Filtration<Int, Real, L>& fil)
{
    TargetMatching<L, Real> result;
    const auto& simplices = fil.simplices();

    for(auto&& kv: diagram_to_values) {
        auto dgm_point = kv.first;
        auto target_point = kv.second;

        auto birth_cvl = simplices[dgm_point.birth].critical_value_location_;
        auto death_cvl = simplices[dgm_point.death].critical_value_location_;

        result[death_cvl] = target_point.death;

        // special case: Vietoris--Rips, dim 0 -- all vertices have critical value 0
        // do not use R column at all
        if (d == 0 and std::is_same<L, VREdge>::value)
            continue;

        result[birth_cvl] = target_point.birth;
    }

    return result;
}
// given target points (diagram_to_values), compute values on intermediate simplices from R, V columns
template<class Int, class Real, class L>
TargetMatching<L, Real> get_target_values(dim_type d, const DiagramToValues<Real>& diagram_to_values, const oineus::Filtration<Int, Real, L>& fil, const oineus::SparseMatrix<Int>& rv_matrix)
{
    TargetMatching<L, Real> result;

    const auto& simplices = fil.simplices();

    std::unordered_map<Int, Real> simplices_to_values;

    for(auto&& kv: diagram_to_values) {
        auto dgm_point = kv.first;
        auto target_point = kv.second;

        size_t birth_idx = dgm_point.birth;
        size_t death_idx = dgm_point.death;

        Real target_birth = target_point.birth;
        Real target_death = target_point.death;

        Real current_birth = simplices[birth_idx].value();
        Real current_death = simplices[death_idx].value();

        Real min_birth = fil.min_value(d);

        auto birth_column = rv_matrix.data[death_idx];
        auto death_column = rv_matrix.v_data[death_idx];

        // birth simplices are in R column
        std::vector<Real> current_r_values;
        std::vector<Int> r_simplex_indices;

        for(auto sigma_idx: birth_column) {
            r_simplex_indices.push_back(sigma_idx);
            current_r_values.push_back(simplices[sigma_idx].value());
        }

        auto target_r_values = lin_interp<Real>(current_r_values, min_birth, current_birth, min_birth, target_birth);

       // death simplices are in V column
        std::vector<Real> current_v_values;
        std::vector<Int> v_simplex_indices;

        for(auto sigma_idx: death_column) {
            v_simplex_indices.push_back(sigma_idx);
            current_v_values.push_back(simplices[sigma_idx].value());
        }

        auto target_v_values = lin_interp<Real>(current_v_values, current_birth, current_death, target_birth, target_death);

        assert(birth_column.size() == target_r_values.size() and target_r_values.size() == r_simplex_indices.size());
        assert(death_column.size() == target_v_values.size() and target_v_values.size() == v_simplex_indices.size());

        if (d == 0 and std::is_same<L, VREdge>::value) {
            // special case: Vietoris--Rips, dim 0 -- all vertices have critical value 0
            // do not use R column at all
            r_simplex_indices.clear();
            target_r_values.clear();
        }

        // put r_ and v_ indices and values together

        std::vector<Int> simplex_indices(std::move(r_simplex_indices));
        simplex_indices.reserve(birth_column.size() + death_column.size());
        simplex_indices.insert(simplex_indices.end(), v_simplex_indices.begin(), v_simplex_indices.end());

        std::vector<Real> target_values(std::move(target_r_values));
        target_values.reserve(birth_column.size() + death_column.size());
        target_values.insert(target_values.end(), target_v_values.begin(), target_v_values.end());

        for(size_t idx = 0; idx < target_values.size(); ++idx) {
            const auto& sigma = simplices[simplex_indices[idx]];
            auto cvl = sigma.critical_value_location_;

            if (result.count(cvl)) {
                Real current_delta = abs(result[cvl] - sigma.value());
                Real new_delta = abs(target_values[idx] - sigma.value());
                // another column wants this simplex to make larger step -> it wins, do not overwrite with our target value
                if (new_delta <= current_delta)
                    continue;
            }

            result[cvl] = target_values[idx];
        }
    }

    return result;
}
} // namespace oineus


#endif //OINEUS_LOSS_H
