#ifndef OINEUS_LOSS_H
#define OINEUS_LOSS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <cmath>

#include "icecream/icecream.hpp"

#include "diagram.h"
#include "timer.h"
#include "filtration.h"
#include "matrix.h"

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
    for(auto x_iter = v.begin(); x_iter != v.end(); ) {
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
typename Diagrams<Real>::Point enhance_point(Real birth, Real death, Real min_birth=-std::numeric_limits<Real>::infinity(), Real max_death=std::numeric_limits<Real>::infinity())
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
using DiagramToValues = std::unordered_map<DgmPoint < size_t>, DgmPoint<Real>>;

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
void match_diagonal_points(const typename oineus::Filtration<Int, Real, L>& current_fil,
        const typename Diagrams<size_t>::Dgm& current_index_dgm,
        typename Diagrams<Real>::Dgm& template_dgm,
        typename hera::AuctionParams<Real>& hera_params,
        DiagramToValues<Real>& result)
{
    using Diagram = typename Diagrams<Real>::Dgm;
    // diagonal point with id
    using DiagP = std::tuple<Real, size_t>;
    using VecDiagP = std::vector<DiagP>;
    VecDiagP current_diagonal_points;

    for(hera::id_type current_dgm_id = 0; current_dgm_id < static_cast<hera::id_type>(current_index_dgm.size()); ++current_dgm_id) {
        if (current_index_dgm[current_dgm_id].is_inf())
            continue;

        size_t birth_idx = current_index_dgm[current_dgm_id].birth;
        size_t death_idx = current_index_dgm[current_dgm_id].death;

        Real birth_val = current_fil.value_by_sorted_id(birth_idx);
        Real death_val = current_fil.value_by_sorted_id(death_idx);

        if (birth_val == death_val)
            current_diagonal_points.emplace_back(birth_val, current_dgm_id);
    }

    // get unmatched template points
    Diagram unmatched_template_points;
    for(auto curr_template: hera_params.matching_b_to_a_) {
        auto current_id = curr_template.first;
        auto template_id = curr_template.second;

        if (current_id < 0 and template_id >= 0)
            unmatched_template_points.emplace_back(template_dgm[template_id]);
    }

    if (unmatched_template_points.size() < current_diagonal_points.size())
        throw std::runtime_error("Not implemented");

    if (unmatched_template_points.size() > current_diagonal_points.size()) {
        // keep most persistent points, operator < for points sorts by persistence first
        std::sort(unmatched_template_points.begin(), unmatched_template_points.end(), std::greater<DgmPoint<Real>>());
        unmatched_template_points.resize(current_diagonal_points.size());
    }

    std::vector<std::tuple<Real, size_t>> diag_unmatched_template;

    for(auto p: unmatched_template_points)
        diag_unmatched_template.emplace_back((p.birth + p.death) / 2, p.id);

    hera_params.clear_matching();

    hera::ws::get_one_dimensional_cost(diag_unmatched_template, current_diagonal_points, hera_params);

    for(auto curr_template: hera_params.matching_b_to_a_) {
        auto current_id = curr_template.first;
        auto template_id = curr_template.second;

        if (current_id < 0 or template_id < 0)
            throw std::runtime_error("negative ids in one-dimensional call");

        result[current_index_dgm.at(current_id)] = template_dgm.at(template_id);
    }
}

template<class Int, class Real, class L>
DiagramToValues<Real> get_barycenter_target(const Filtration<Int, Real, L>& fil, VRUDecomposition<Int>& rv, dim_type d)
{
    DiagramToValues<Real> result;
    constexpr bool is_vr = std::is_same_v<VREdge, L>;

    auto index_dgm = rv.index_diagram(fil, false, false).get_diagram_in_dimension(d);
    auto dgm = rv.diagram(fil, false).get_diagram_in_dimension(d);

    if (dgm.size()) {
        Real avg_birth = (is_vr ? 0 : std::accumulate(dgm.begin(), dgm.end(), static_cast<Real>(0), [&](auto x, auto p) { return x + p.birth; })) / dgm.size();
        Real avg_death = std::accumulate(dgm.begin(), dgm.end(), static_cast<Real>(0), [&](auto x, auto p) { return x + p.death; }) / dgm.size();

        for(auto p: index_dgm)
            if (fil.value_by_sorted_id(p.birth) != fil.value_by_sorted_id(p.death))
                result[p] = {avg_birth, avg_death};
    }

    return result;
}

template<class Int, class Real, class L>
DiagramToValues<Real> get_bruelle_target(const Filtration<Int, Real, L>& current_fil,
        VRUDecomposition<Int>& rv,
        int p,
        int q,
        int i_0,
        dim_type d,
        bool minimize,
        Real min_birth = -std::numeric_limits<Real>::infinity(),
        Real max_death = std::numeric_limits<Real>::infinity())
{
    if (q == 0 and p == 2)
        return get_bruelle_target_2_0(current_fil, rv, i_0, d, minimize, min_birth, max_death);
    else
        throw std::runtime_error("Not implemented");
}

template<class Int, class Real, class L>
DiagramToValues<Real> get_bruelle_target_2_0(const Filtration<Int, Real, L>& current_fil,
        VRUDecomposition<Int>& rv,
        int n_keep,
        dim_type d,
        bool minimize,
        Real min_birth,
        Real max_death)
{
    constexpr bool is_vr = std::is_same_v<L, VREdge>;
    DiagramToValues<Real> result;

    Real epsilon = get_nth_persistence(current_fil, rv, d, n_keep);

    // false flags: no infinite points, no points with zero persistence
    auto index_dgm = rv.index_diagram(current_fil, false, false).get_diagram_in_dimension(d);

    for(auto p: index_dgm) {
        Real birth_val = current_fil.value_by_sorted_id(p.birth);
        Real death_val = current_fil.value_by_sorted_id(p.death);
        if (abs(death_val - birth_val) <= epsilon) {
            result[p] = minimize ? denoise_point(birth_val, death_val, DenoiseStrategy::Midway) : enhance_point(birth_val, death_val, min_birth, max_death);
        }
    }

    return result;
}

// if point is in quadrant defined by (t, t),
// move it to horizontal or vertical quadrant border, whichever is closer
template<class Int, class Real, class L>
DiagramToValues<Real> get_well_group_target(dim_type  d,
        const Filtration<Int, Real, L>& current_fil,
        VRUDecomposition<Int>& rv,
        Real t)
{
    DiagramToValues<Real> result;

    constexpr bool is_vr = std::is_same_v<L, VREdge>;

    auto index_dgm = rv.index_diagram(current_fil, false, false).get_diagram_in_dimension(d);

    for(auto p: index_dgm) {
        Real birth_val = current_fil.value_by_sorted_id(p.birth);
        Real death_val = current_fil.value_by_sorted_id(p.death);

        // check if in quadrant
        if (current_fil.negate() and (birth_val <= t or death_val >= t))
            continue;
        if (not current_fil.negate() and (birth_val >= t or death_val <= t))
            continue;

        Real target_birth, target_death;

        if (abs(birth_val - t) < abs(death_val - t)) {
            target_birth = t;
            target_death = death_val;
        } else {
            target_birth = birth_val;
            target_death = t;
        }

        result[p] = { target_birth, target_death };
    }

    return result;
}

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

template<class Int, class Real, class L>
DiagramToValues<Real> get_target_from_matching(typename Diagrams<Real>::Dgm& template_dgm,
        const Filtration<Int, Real, L>& current_fil,
        VRUDecomposition<Int>& rv,
        dim_type d,
        Real wasserstein_q,
        bool match_inf_points,
        bool match_diag_points)
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
    hera_params.match_inf_points = match_inf_points;
    hera_params.wasserstein_power = wasserstein_q;

    auto current_index_dgm = rv.index_diagram(current_fil, false, false).get_diagram_in_dimension(d);

    Diagram current_dgm;
    current_dgm.reserve(current_index_dgm.size());

    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = std::numeric_limits<Real>::lowest();

    for(auto&& p : template_dgm) {
        min_val = std::min(min_val, p.birth);
        min_val = std::min(min_val, p.death);
        max_val = std::max(max_val, p.birth);
        max_val = std::max(max_val, p.death);
    }

    if (min_val > max_val or min_val == std::numeric_limits<Real>::max() or min_val == std::numeric_limits<Real>::infinity()
        or max_val == std::numeric_limits<Real>::lowest() or max_val == std::numeric_limits<Real>::infinity()) {
        throw std::runtime_error("bad max/min value");
    }

    for(hera::id_type current_dgm_id = 0; current_dgm_id < current_index_dgm.size(); ++current_dgm_id) {

        auto birth_idx = current_index_dgm[current_dgm_id].birth;
        auto death_idx = current_index_dgm[current_dgm_id].death;

        auto birth_val = current_fil.value_by_sorted_id(birth_idx);
        auto death_val = current_fil.value_by_sorted_id(death_idx);

        // do not include diagonal points, save them in a separate vector
        if (birth_val != death_val)
            current_dgm.emplace_back(birth_val, death_val, current_dgm_id);
    }

    // template_dgm: bidders, a
    // current_dgm: items, b
    hera::wasserstein_cost<Diagram>(template_dgm, current_dgm, hera_params);

    for(auto curr_template: hera_params.matching_b_to_a_) {
        auto current_id = curr_template.first;
        auto template_id = curr_template.second;

        if (current_id < 0)
            continue;

        if (template_id >= 0)
            // matched to off-diagonal point of template diagram
            result[current_index_dgm.at(current_id)] = template_dgm.at(template_id);
        else {
            // point must disappear, move to (birth, birth), clamp, if necessary
            auto target_point = current_dgm.at(current_id);
            target_point.death = target_point.birth;
            if (current_fil.negate()) {
                if (target_point.death > max_val)
                    target_point.birth = target_point.death = max_val;
                else if (target_point.birth < min_val)
                    target_point.birth = target_point.death = min_val;
            } else {
                if (target_point.birth > max_val)
                    target_point.birth = target_point.death = max_val;
                else if (target_point.death < min_val)
                    target_point.birth = target_point.death = min_val;
            }
            result[current_index_dgm.at(current_id)] = target_point;
        }
    }

    if (match_diag_points)
        match_diagonal_points(current_fil, current_index_dgm, template_dgm, hera_params, result);

    return result;
}

// return the n-th (finite) persistence value in dimension d
// points at infinity are ignored
template<class Int, class Real, class L>
Real get_nth_persistence(const Filtration<Int, Real, L>& fil, const VRUDecomposition<Int>& rv_matrix, dim_type d, int n)
{
    if (n < 1) {
        throw std::runtime_error("get_nth_persistence: n must be at least 1");
    }

    auto index_diagram = rv_matrix.template index_diagram<Real, L>(fil, false, false).get_diagram_in_dimension(d);
    std::vector<Real> ps;
    ps.reserve(index_diagram.size());

    for(auto p: index_diagram) {
        Real birth = fil.simplices()[p.birth].value();
        Real death = fil.simplices()[p.death].value();
        Real pers = abs(death - birth);
        ps.push_back(pers);
    }

    Real result {0};
    if (ps.size() >= n) {
        std::nth_element(ps.begin(), ps.begin() + n - 1, ps.end(), std::greater<Real>());
        result = ps[n - 1];
    }

    return result;
}

// points (b, d) with persistence | b - d| <= eps should go to (b, b)
template<class Int, class Real, class L>
DiagramToValues<Real> get_denoise_target(dim_type d, const Filtration<Int, Real, L>& fil, const VRUDecomposition<Int>& rv_matrix, Real eps, DenoiseStrategy strategy)
{
    DiagramToValues<Real> result;
    auto index_diagram = rv_matrix.template index_diagram<Real, L>(fil, false, false)[d];

    for(auto p: index_diagram) {
        Real birth = fil.value_by_sorted_id(p.birth);
        Real death = fil.value_by_sorted_id(p.death);
        Real pers = abs(death - birth);
        if (pers <= eps)
            result[p] = denoise_point(birth, death, strategy);
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
TargetMatching<L, Real> get_target_values(dim_type d, const DiagramToValues<Real>& diagram_to_values, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& rv_matrix)
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

        auto birth_column = rv_matrix.r_data[death_idx];
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


template<class Int, class Real, class L>
std::vector<Int> increase_birth_x(dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_birth)
{
    if (not decmp.dualize())
        throw std::runtime_error("expected cohomology");

    if (not fil.cmp(fil.simplices()[positive_simplex_idx].value(), target_birth))
        throw std::runtime_error("target_birth cannot preceed current value");

    std::vector<Int> result;

    auto& v_col = decmp.v_data.at(fil.index_in_matrix(positive_simplex_idx, decmp.dualize()));

    for(auto index_in_matrix = v_col.rbegin(); index_in_matrix != v_col.rend(); ++index_in_matrix) {
        auto fil_idx = fil.index_in_filtration(*index_in_matrix, decmp.dualize());
        const auto& sigma = fil.simplices().at(fil_idx);

        if (fil.cmp(target_birth, sigma.value()))
            break;

        result.push_back(fil_idx);
    }

    if (result.empty())
        throw std::runtime_error("increase_birth_x: empty");

    return result;
}

template<class Int, class Real, class L>
std::vector<Int> decrease_birth_x(dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_birth)
{
    if (not decmp.dualize())
        throw std::runtime_error("expected cohomology");

    if (not fil.cmp(target_birth, fil.simplices()[positive_simplex_idx].value()))
        throw std::runtime_error("target_birth cannot preceed current value");

    std::vector<Int> result;

//    std::cerr << "in decrease_birth_x: u_data_t.size = " << decmp.u_data_t.size() << ", positive_simplex_idx = " << positive_simplex_idx << ", index = " << fil.index_in_matrix(positive_simplex_idx, decmp.dualize()) << ", row.size = " << decmp.u_data_t.at(fil.index_in_matrix(positive_simplex_idx, decmp.dualize())).size() << std::endl;

    for(auto index_in_matrix : decmp.u_data_t.at(fil.index_in_matrix(positive_simplex_idx, decmp.dualize()))) {
        auto fil_idx = fil.index_in_filtration(index_in_matrix, decmp.dualize());
        const auto& sigma = fil.simplices()[fil_idx];

//        std::cerr << "fil_idx = " << fil_idx << ", index_in_matrix = " << index_in_matrix << std::endl;

        if (fil.cmp(sigma.value(), target_birth)) {
//            std::cerr << " breaking for value" << ", target_birth = " << target_birth << ", sigma.value() = " << sigma.value() << std::endl;
            break;
        }

        result.push_back(fil_idx);
    }

    if (result.empty())
        throw std::runtime_error("decrease_birth_x: empty");

    return result;
}

template<class Int, class Real, class L>
std::vector<Int> increase_death_x(dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_death)
{
    std::vector<Int> result;

    const auto& u_rows = decmp.u_data_t;
    const auto& r_cols = decmp.r_data;
    const auto& simplices = fil.simplices();

    size_t n_cols = decmp.v_data.size();
    Int sigma = low(r_cols[negative_simplex_idx]);

    if (not(sigma >= 0 and sigma < r_cols.size()))
        throw std::runtime_error("expected negative simplex");

//    std::cerr << "in increase_death_x: u_data_t.size = " << decmp.u_data_t.size() << ", negative_simplex_idx = " << negative_simplex_idx << ", row.size = " << u_rows[negative_simplex_idx].size() << std::endl;

    for(auto tau_idx: u_rows.at(negative_simplex_idx)) {
        const auto& tau = simplices.at(tau_idx);
        assert(tau.dim() == d);
//        std::cerr << "tau_idx = " << tau_idx;
        if (fil.cmp(target_death, tau.value())) {
//            std::cerr << ", breaking for value" << std::endl;
            break;
        }

        if (low(decmp.r_data[tau_idx]) <= sigma)
            result.push_back(tau_idx);
//        else
//            std::cerr << ", not pushing: low = " << low(decmp.r_data[tau_idx]) << ", sigma = " << sigma << std::endl;
    }

    if (result.empty())
        throw std::runtime_error("increase_death_x: empty");

    return result;
}

template<class Int, class Real, class L>
std::vector<Int> decrease_death_x(dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_death)
{
    std::vector<Int> result;

    auto& r_cols = decmp.r_data;
    Int sigma = low(r_cols[negative_simplex_idx]);

    assert(sigma >= 0 and sigma < r_cols.size());

    auto& v_col = decmp.v_data[negative_simplex_idx];

    for(auto tau_idx_it = v_col.rbegin(); tau_idx_it != v_col.rend(); ++tau_idx_it) {
        auto tau_idx = *tau_idx_it;
        const auto& tau = fil.simplices()[tau_idx];
        assert(tau.dim() == d);

        if (fil.cmp(tau.value(), target_death))
            break;

        // explicit check for is_zero is not necessary for signed Int, low returns -1 for empty columns
        if (low(decmp.r_data[tau_idx]) < sigma or is_zero(decmp.r_data[tau_idx]))
            continue;

        result.push_back(tau_idx);
    }

    assert(result.size());

    return result;
}

template<class Int, class Real, class L>
std::vector<Int> change_death_x(dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_death)
{
    Real current_death = fil.simplices()[negative_simplex_idx].value();
    if (fil.cmp(target_death, current_death))
        return decrease_death_x(d, negative_simplex_idx, fil, decmp, target_death);
    else if (fil.cmp(current_death, target_death))
        return increase_death_x(d, negative_simplex_idx, fil, decmp, target_death);
    else // current_death = target_death, no change required
        return {};
}

template<class Int, class Real, class L>
std::vector<Int> change_birth_x(dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp_coh, Real target_birth)
{
    if (not decmp_coh.dualize())
        throw std::runtime_error("expected cohomology");

    Real current_birth = fil.simplices()[positive_simplex_idx].value();
    if (fil.cmp(target_birth, current_birth))
        return decrease_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
    else if (fil.cmp(current_birth, target_birth))
        return increase_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
    else
        return {};
}

// given target points (diagram_to_values), compute values on intermediate simplices using X set definition
template<class Int, class Real, class L>
TargetMatching<L, Real> get_target_values_x(dim_type d,
        const DiagramToValues<Real>& diagram_to_values,
        const oineus::Filtration<Int, Real, L>& fil,
        const oineus::VRUDecomposition<Int>& decmp_hom,
        const oineus::VRUDecomposition<Int>& decmp_coh)
{

//    std::cerr << "enter get_target_values_x, diagram_to_values.size = " << diagram_to_values.size() << ", fil.size = " << fil.size() << std::endl;

    if (decmp_hom.dualize())
        throw std::runtime_error("this parameter must be homology");

    if (not decmp_coh.dualize())
        throw std::runtime_error("this parameter must be cohomology");

    TargetMatching<L, Real> result;

    const auto& simplices = fil.simplices();

    std::unordered_map<Int, Real> simplices_to_values;

    int n_conflicts = 0;
    int n_single_point_conflicts = 0;
    int n_points_with_conflict = 0;

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

//        std::cerr << "indices: (" << birth_idx << ", " << death_idx <<  "): MOVE (" << current_birth << ", " << current_death << ") -> (" << target_birth << ", " << target_death << ")" << std::endl;

        auto death_x = change_death_x<Int>(d, death_idx, fil, decmp_hom, target_death);
//        if (death_x.size() > 1)
//            std::cerr << "death_x.size = " << death_x.size() << std::endl;

//        for(auto x_d : death_x)
//            std::cerr << fil.simplices()[x_d] << std::endl;

        auto birth_x = change_birth_x<Int>(d, birth_idx, fil, decmp_coh, target_birth);
//        if (birth_x.size() > 1)
//            std::cerr << "birth_x.size = " << birth_x.size() << std::endl;

//        for(auto x_b : birth_x)
//            std::cerr << fil.simplices()[x_b] << std::endl;

        {
            // check for intersection
            std::vector<Int> birth_death;
            std::set_intersection(birth_x.begin(), birth_x.end(), death_x.begin(), death_x.end(), std::back_inserter(birth_death));
            n_points_with_conflict += (not birth_death.empty());
            n_single_point_conflicts += birth_death.size();
        }

        for(auto b_idx: birth_x) {
            const auto& sigma = simplices[b_idx];
            auto cvl = sigma.critical_value_location_;

            n_conflicts += result.count(cvl);

            // another column wants this location to make larger step -> it wins, do not overwrite with our target value
            if (result.count(cvl) and abs(target_birth - sigma.value()) <= abs(result[cvl] - sigma.value()))
                continue;

            result[cvl] = target_birth;
        }

        for(auto d_idx: death_x) {
            const auto& sigma = simplices[d_idx];
            auto cvl = sigma.critical_value_location_;

            n_conflicts += result.count(cvl);

            // another column wants this location to make larger step -> it wins, do not overwrite with our target value
            if (result.count(cvl) and abs(target_death - sigma.value()) <= abs(result[cvl] - sigma.value()))
                continue;

            result[cvl] = target_death;
        }
    }

    if (n_conflicts or n_points_with_conflict or n_single_point_conflicts)
        std::cerr << "conflicts for one critical value location: " << n_conflicts << ", single dgm points with conflict: " << n_points_with_conflict << ", conflicts from single point: " << n_single_point_conflicts << std::endl;

    return result;
}

} // namespace oineus


#endif //OINEUS_LOSS_H
