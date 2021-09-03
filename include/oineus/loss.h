#ifndef OINEUS_LOSS_H
#define OINEUS_LOSS_H

#include <vector>
#include <cassert>
#include <unordered_map>
#include <cmath>

#include "diagram.h"
#include "filtration.h"
#include "matrix.h"

#pragma once

namespace oineus {

template<class Real>
using DiagramToValues = std::unordered_map<oineus::DgmPoint<size_t>, oineus::DgmPoint<Real>>;

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

//template<class Int, class Real, class L>
//typename DiagramToValues<Real> diagram_to_values(dim_type d, const oineus::Filtration<Int, Real, L>& fil, const oineus::SparseMatrix<Int>& rv_matrix, Real eps)
//{
//    auto index_diagram = rv_matrix.template index_diagram_finite<Real, L>(fil)[d];
//    auto& simplices = fil.simplices(d);
//
//    for(auto p : index_diagram) {
//        auto birth_simplex = simplices[fil.adjust_index(p.birth, d)];
//    }
//}

template<class Int, class Real, class L>
TargetMatching<L, Real> get_target_values(dim_type d, const DiagramToValues<Real>& diagram_to_values, const oineus::Filtration<Int, Real, L>& fil, const oineus::SparseMatrix<Int>& rv_matrix)
{
    TargetMatching<L, Real> result;

    const auto& simplices = fil.simplices();

    const bool negate = fil.negate();

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

        auto birth_column = rv_matrix.data[birth_idx];
        auto death_column = rv_matrix.v_data[birth_idx];

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
