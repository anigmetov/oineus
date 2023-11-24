#pragma once

#include <vector>
#include <utility>
#include <type_traits>

#include "filtration.h"
#include "inclusion_filtration.h"
#include "decomposition.h"
//#include "loss.h"

namespace oineus {

struct ComputeFlags {
    bool compute_cohomology {false};
    bool compute_homology_u {false};
    bool compute_cohomology_u {false};
};

inline std::ostream& operator<<(std::ostream& out, const ComputeFlags& f)
{
    out << "ComputeFlags(compute_cohomology = " << (f.compute_cohomology ? "True" : "False");
    out << ", compute_homology_u = " << (f.compute_homology_u ? "True" : "False");
    out << ", compute_cohomology_u = " << (f.compute_cohomology_u ? "True)" : "False)");
    return out;
}

template<class Cell_, class Real_>
class TopologyOptimizer {
public:

    static_assert(std::is_floating_point_v<Real_>, "Real_ must be floating point type");

    using Fil = Filtration<Cell_, Real_>;
    using Cell = CellWithValue<Cell_, Real_>;
    using Real = typename Cell::Real;
    using Int = typename Cell::Int;
    using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;
    using Indices = std::vector<Int>;
    using Values = std::vector<Real>;
    using DgmTarget = std::unordered_map<DgmPoint<Real>, DgmPoint<Real>>;
    using CriticalSet = std::pair<Real, Indices>;
    using CriticalSets = std::vector<CriticalSet>;

    using Decomposition = VRUDecomposition<Int>;
    using Dgms = Diagrams<Real>;
    using Dgm = typename Dgms::Dgm;

    struct SimplexTarget {
        Real current_value;
        Real target_value;
        bool is_positive;

        bool increase_birth(bool negate) const
        {
            if (not is_positive)
                return false;
            if (negate)
                return target_value < current_value;
            else
                return target_value > current_value;
        }

        bool decrease_birth(bool negate) const
        {
            if (not is_positive)
                return false;
            if (negate)
                return target_value > current_value;
            else
                return target_value < current_value;
        }

        bool increase_death(bool negate) const
        {
            if (is_positive)
                return false;
            if (negate)
                return target_value < current_value;
            else
                return target_value > current_value;
        }

        bool decrease_death(bool negate) const
        {
            if (is_positive)
                return false;
            if (negate)
                return target_value > current_value;
            else
                return target_value < current_value;
        }
    };

    using Target = std::unordered_map<size_t, SimplexTarget>;

    struct IndicesValues {
        Indices indices;
        Values values;

        void push_back(size_t i, Real v)
        {
            indices.push_back(i);
            values.push_back(v);
        }

        void emplace_back(size_t i, Real v)
        {
            indices.emplace_back(i);
            values.emplace_back(v);
        }

        friend std::ostream& operator<<(std::ostream& out, const IndicesValues& iv)
        {
            out << "IndicesValue(indices=";
            out << container_to_string(iv.indices);
            out << ", values=";
            out << container_to_string(iv.values);
            return out;
        }
    };

//    TopologyOptimizer(const BoundaryMatrix& boundary_matrix, const Values& values, bool negate = false)
//            :
//            decmp_hom_(boundary_matrix),
//            negate_(negate)
//    {
//    }

    TopologyOptimizer(const Fil& fil)
            :
            decmp_hom_(fil, false),
            decmp_coh_(fil, true),
            fil_(fil),
            negate_(fil.negate())
    {
        params_hom_.clearing_opt = false;
        params_coh_.clearing_opt = false;
    }

    TopologyOptimizer(const Fil& fil, const ComputeFlags& hints)
            :
            decmp_hom_(fil, false),
            decmp_coh_(fil, true),
            fil_(fil),
            negate_(fil.negate())
    {
        params_hom_.compute_u = hints.compute_homology_u;
        params_coh_.compute_u = hints.compute_cohomology_u;

        if (hints.compute_homology_u)
            params_hom_.clearing_opt = false;
        if (hints.compute_cohomology_u)
            params_coh_.clearing_opt = false;
    }

    bool cmp(Real a, Real b) const
    {
        if (negate_)
            return a > b;
        else
            return a < b;
    }

    ComputeFlags get_flags(const Target& target)
    {
        bool increase_birth = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.increase_birth(negate_); });
        bool decrease_birth = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.decrease_birth(negate_); });
        bool increase_death = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.increase_death(negate_); });
        bool decrease_death = std::accumulate(target.begin(), target.end(), false,
                [this](bool x, auto kv) { return x or kv.second.decrease_death(negate_); });

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_birth;
        result.compute_homology_u = increase_death;
        result.compute_cohomology_u = decrease_birth;

        return result;
    }

    ComputeFlags get_flags(const DgmTarget& target)
    {
        bool increase_birth = false;
        bool increase_death = false;
        bool decrease_birth = false;
        bool decrease_death = false;

        for(const auto& [point, target_point]: target) {

            if (cmp(point.birth, target_point.birth)) {
                increase_birth = true;
            } else if (cmp(target_point.birth, point.birth)) {
                decrease_birth = true;
            }

            if (cmp(point.death, target_point.death)) {
                increase_death = true;
            } else if (cmp(target_point.death, point.death)) {
                decrease_death = true;
            }
        }

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_birth;
        result.compute_homology_u = increase_death;
        result.compute_cohomology_u = decrease_birth;

        return result;
    }

    ComputeFlags get_flags(const Indices& indices, const Values& values)
    {
        bool increase_birth = false;
        bool increase_death = false;
        bool decrease_birth = false;
        bool decrease_death = false;

        for(size_t i = 0 ; i < indices.size() ; ++i) {
            auto simplex_idx = indices[i];
            Real current_value = fil_.get_cell_value(simplex_idx);
            Real target_value = values[i];
            bool is_positive = decmp_hom_.is_positive(simplex_idx);

            if (is_positive and cmp(current_value, target_value)) {
                increase_birth = true;
            } else if (is_positive and cmp(target_value, current_value)) {
                decrease_birth = true;
            } else if (not is_positive and cmp(current_value, target_value)) {
                increase_death = true;
            } else if (not is_positive and cmp(target_value, current_value)) {
                decrease_death = true;
            }
        }

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_birth;
        result.compute_homology_u = increase_death;
        result.compute_cohomology_u = decrease_birth;

        return result;
    }

    dim_type get_dimension(size_t simplex_index) const
    {
        if (fil_.size())
            return fil_.dim_by_sorted_id(simplex_index);
        else
            return 0;
    }

    CriticalSet singleton(size_t index, Real value)
    {
        if (!decmp_hom_.is_reduced or (params_hom_.compute_u and not decmp_hom_.has_matrix_u())) {
            decmp_hom_.reduce_serial(params_hom_);
        }

        if (decmp_hom_.is_negative(index)) {
            return {value, change_death(index, value)};
        } else {
            return {value, change_birth(index, value)};
        }
    }

    CriticalSets singletons(const Indices& indices, const Values& values)
    {
        if (indices.size() != values.size())
            throw std::runtime_error("indices and values must have the same size");

        auto flags = get_flags(indices, values);
        //IC(flags);
        params_coh_.compute_u = flags.compute_cohomology_u;
        params_hom_.compute_u = flags.compute_homology_u;

        CriticalSets result;
        result.reserve(indices.size());

        for(size_t i = 0 ; i < indices.size() ; ++i) {
            result.emplace_back(singleton(indices[i], values[i]));
        }

        return result;
    }

    void update(const Values& new_values, int n_threads = 1)
    {
        fil_.update(new_values);

        decmp_hom_ = Decomposition(fil_, false);
        decmp_coh_ = Decomposition(fil_, true);
    }

    decltype(auto) convert_critical_sets(const CriticalSets& critical_sets) const
    {
        std::unordered_map<size_t, Values> result;
        for(const auto& crit_set: critical_sets) {
            Real value = crit_set.first;
            for(size_t index: crit_set.second) {
                result[index].push_back(value);
            }
        }
        return result;
    }

    Real get_cell_value(size_t simplex_idx) const
    {
        return fil_.value_by_sorted_id(simplex_idx);
    }

//    Target dgm_target_to_target(const DgmTarget& dgm_target) const
//    {
//        Target target;
//
//        for(auto&& [point, target_point]: dgm_target) {
//            size_t birth_simplex = point.birth;
//            Real current_birth_value = get_cell_value(birth_simplex);
//            Real target_birth_value = target_point.birth;
//
//            if (point.birth != target_point.birth)
//                target.emplace(point.birth_index, {current_birth_value, target_birth_value, true});
//
//            size_t death_simplex = point.death;
//            Real current_death_value = get_cell_value(death_simplex);
//            Real target_death_value = target_point.death;
//
//            if (current_death_value != target_death_value)
//                target.emplace(death_simplex, {current_death_value, target_death_value, false});
//        }
//
//        return target;
//    }

    IndicesValues simplify(Real epsilon, DenoiseStrategy strategy, dim_type dim)
    {
        if (not decmp_hom_.is_reduced)
            decmp_hom_.reduce_serial(params_hom_);

        IndicesValues result;

        auto dgm = decmp_hom_.diagram(fil_, false)[dim];

//        causes bugs: spurious points in diagram, need to materialize the diagram
//        for(auto p: decmp_hom_.diagram(fil_, false)[dim]) {
        for(auto p: dgm) {
            if (p.birth_index == p.death_index)
                throw std::runtime_error("bad p in simplify");
            if (p.persistence() <= epsilon) {
                if (strategy == DenoiseStrategy::BirthBirth) {
                    result.push_back(p.death_index, p.birth);
                } else if (strategy == DenoiseStrategy::DeathDeath)
                    result.push_back(p.birth_index, p.death);
                else if (strategy == DenoiseStrategy::Midway) {
                    result.push_back(p.birth_index, (p.birth + p.death) / 2);
                    result.push_back(p.death_index, (p.birth + p.death) / 2);
                }
            }
        }

        return result;
    }

    Real get_nth_persistence(dim_type d, int n)
    {
        if (!decmp_hom_.is_reduced)
            decmp_hom_.reduce(this->params_hom_);

        return oineus::get_nth_persistence(fil_, decmp_hom_, d, n);
    }

    std::pair<IndicesValues, Real> match_and_distance(typename Diagrams<Real>::Dgm& template_dgm, dim_type d, Real wasserstein_q)
    {
        // set ids in template diagram
        for(auto i = 0 ; i < template_dgm.size() ; ++i) {
            template_dgm[i].id = i;

            if (template_dgm[i].is_inf())
                throw std::runtime_error("infinite point in template diagram");
        }

        using Diagram = typename Diagrams<Real>::Dgm;

        IndicesValues result;

        hera::AuctionParams<Real> hera_params;
        hera_params.return_matching = true;
        hera_params.match_inf_points = false;
        hera_params.wasserstein_power = wasserstein_q;

        if (not decmp_hom_.is_reduced)
            decmp_hom_.reduce(params_hom_);

        Diagram current_dgm = decmp_hom_.diagram(fil_, false).get_diagram_in_dimension(d);

        for(auto i = 0 ; i < current_dgm.size() ; ++i) {
            current_dgm[i].id = i;
        }

        // template_dgm: bidders, a
        // current_dgm: items, b
        auto hera_res = hera::wasserstein_cost_detailed<Diagram>(template_dgm, current_dgm, hera_params);

        for(auto curr_template: hera_res.matching_b_to_a_) {
            auto current_id = curr_template.first;
            auto template_id = curr_template.second;

            if (current_id < 0)
                continue;

            size_t birth_idx = current_dgm.at(current_id).birth_index;
            size_t death_idx = current_dgm.at(current_id).death_index;

            Real birth_target;
            Real death_target;

            if (template_id >= 0) {
                // matched to off-diagonal point of template diagram

                birth_target = template_dgm.at(template_id).birth;
                death_target = template_dgm.at(template_id).death;
            } else {
                // matched to diagonal point of template diagram
                auto curr_proj_id = -template_id - 1;
                Real m = (current_dgm.at(curr_proj_id).birth + current_dgm.at(curr_proj_id).death) / 2;
                birth_target = death_target = m;
            }

            result.push_back(birth_idx, birth_target);
            result.push_back(death_idx, death_target);
        }

        return {result, hera_res.distance};
    }

    IndicesValues match(typename Diagrams<Real>::Dgm& template_dgm, dim_type d, Real wasserstein_q)
    {
        return match_and_distance(template_dgm, d, wasserstein_q).first;
    }

    IndicesValues combine_loss(const CriticalSets& critical_sets, ConflictStrategy strategy)
    {
        if (strategy != ConflictStrategy::FixCritAvg)
            return combine_loss(critical_sets, Target(), strategy);
        else
            throw std::runtime_error("Need target to use FixCritAvg strategy");
    }

    IndicesValues combine_loss(const CriticalSets& critical_sets, const Target& target, ConflictStrategy strategy)
    {
        CALI_CXX_MARK_FUNCTION;
        auto simplex_to_values = convert_critical_sets(critical_sets);
        IndicesValues indvals;

        if (strategy == ConflictStrategy::Max) {
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                Real current_value = get_cell_value(simplex_idx);
                // compare by displacement from current value
                Real target_value = *std::max_element(values.begin(), values.end(), [current_value](Real a, Real b) { return abs(a - current_value) < abs(b - current_value); });
                indvals.push_back(simplex_idx, target_value);
            }
        } else if (strategy == ConflictStrategy::Avg) {
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                Real target_value = std::accumulate(values.begin(), values.end(), static_cast<Real>(0)) / values.size();
                indvals.emplace_back(simplex_idx, target_value);
            }
        } else if (strategy == ConflictStrategy::Sum) {
            // return all prescribed values, gradient of loss will be summed
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                for(auto value: values) {
                    indvals.emplace_back(simplex_idx, value);
                }
            }
        } else if (strategy == ConflictStrategy::FixCritAvg) {
            // send critical cells according to the matching loss
            // average on others
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                // where matching loss tells critical cells to go
                // is contained in critical_prescribed map
                auto critical_iter = target.find(simplex_idx);
                Real target_value;
                if (critical_iter == target.end())
                    target_value = std::accumulate(values.begin(), values.end(), static_cast<Real>(0)) / values.size();
                else
                    target_value = critical_iter->second.target_value;

                indvals.emplace_back(simplex_idx, target_value);
            }
        }

        return indvals;
    }

    IndicesValues combine_loss(const Indices& indices, const Values& values, ConflictStrategy strategy)
    {
        return combine_loss(singletons(indices, values), strategy);
    }

    Dgms compute_diagram(bool include_inf_points)
    {
        if (!decmp_hom_.is_reduced)
            decmp_hom_.reduce(params_hom_);

        return decmp_hom_.diagram(fil_, include_inf_points);
    }

    void reduce_all()
    {
        params_hom_.clearing_opt = false;
        params_hom_.compute_u = params_hom_.compute_v = true;
        if (!decmp_hom_.is_reduced or (params_hom_.compute_u and not decmp_hom_.has_matrix_u())) {
            decmp_hom_.reduce_serial(params_hom_);
        }

        params_coh_.clearing_opt = false;
        params_coh_.compute_u = params_coh_.compute_v = true;
        if (!decmp_coh_.is_reduced or (params_coh_.compute_u and not decmp_coh_.has_matrix_u())) {
            decmp_coh_.reduce_serial(params_coh_);
        }
    }

    Indices increase_birth(size_t positive_simplex_idx, Real target_birth) const
    {
        if (not fil_.cmp(fil_.get_cell_value(positive_simplex_idx), target_birth))
            throw std::runtime_error("target_birth cannot precede current value");

        Indices result;

        auto& v_col = decmp_coh_.v_data.at(fil_.index_in_matrix(positive_simplex_idx, true));

        for(auto index_in_matrix = v_col.rbegin() ; index_in_matrix != v_col.rend() ; ++index_in_matrix) {
            auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
            if (fil_.cmp(target_birth, fil_.get_cell_value(fil_idx)))
                break;

            result.push_back(fil_idx);
        }

        if (result.empty())
            throw std::runtime_error("increase_birth: empty");

        return result;
    }

    Indices increase_birth(size_t positive_simplex_idx) const
    {
        return increase_birth(positive_simplex_idx, fil_.infinity());
    }

    Indices decrease_birth(size_t positive_simplex_idx, Real target_birth) const
    {
        if (not fil_.cmp(target_birth, fil_.get_cell_value(positive_simplex_idx)))
            throw std::runtime_error("target_birth cannot precede current value");

        Indices result;

        for(auto index_in_matrix: decmp_coh_.u_data_t.at(fil_.index_in_matrix(positive_simplex_idx, true))) {
            auto fil_idx = fil_.index_in_filtration(index_in_matrix, true);

            if (fil_.cmp(fil_.get_cell_value(fil_idx), target_birth)) {
                break;
            }

            result.push_back(fil_idx);
        }

        if (result.empty())
            throw std::runtime_error("decrease_birth: empty");

        return result;
    }

    Indices decrease_birth(size_t positive_simplex_idx) const
    {
        return decrease_birth(positive_simplex_idx, -fil_.infinity());
    }

    Indices increase_death(size_t negative_simplex_idx, Real target_death) const
    {
        Indices result;

        const auto& u_rows = decmp_hom_.u_data_t;
        const auto& r_cols = decmp_hom_.r_data;

        size_t n_cols = decmp_hom_.v_data.size();
        Int sigma = low(r_cols[negative_simplex_idx]);

        if (not(sigma >= 0 and sigma < r_cols.size()))
            throw std::runtime_error("expected negative simplex");

        for(auto tau_idx: u_rows.at(negative_simplex_idx)) { // loop over all indices in this row
            if (fil_.cmp(target_death, fil_.get_cell_value(tau_idx))) {
                break;
            }

            // if (low(decmp_hom_.r_data[tau_idx]) <= sigma) { // do not need this
            result.push_back(tau_idx);
            // }
        }

        if (result.empty())
            throw std::runtime_error("increase_death: empty");

        return result;
    }

    Indices increase_death(size_t negative_simplex_idx) const
    {
        return increase_death(negative_simplex_idx, fil_.infinity());
    }

    Indices decrease_death(size_t negative_simplex_idx, Real target_death) const
    {
        Indices result;

        auto& r_cols = decmp_hom_.r_data;
        Int sigma = low(r_cols[negative_simplex_idx]);

        if (not(sigma >= 0 and sigma < r_cols.size()))
            throw std::runtime_error("expected negative simplex");

        auto& v_col = decmp_hom_.v_data[negative_simplex_idx];

        for(auto tau_idx_it = v_col.rbegin() ; tau_idx_it != v_col.rend() ; ++tau_idx_it) {
            auto tau_idx = *tau_idx_it;

            if (fil_.cmp(fil_.get_cell_value(tau_idx), target_death))
                break;

            // explicit check for is_zero is not necessary for signed Int, low returns -1 for empty columns
            if (low(decmp_hom_.r_data[tau_idx]) < sigma or is_zero(decmp_hom_.r_data[tau_idx]))
                continue;

            result.push_back(tau_idx);
        }

        if (result.empty())
            throw std::runtime_error("decrease_death: empty");

        return result;
    }

    Indices decrease_death(size_t negative_simplex_idx) const
    {
        return decrease_death(negative_simplex_idx, -fil_.infinity());
    }


    /*
        Compute the slope of the linear interpolate function. `death=true` for modifying birth. 
        When modify death, T is the negative value of tau, B is the min value in V[tau]
        When modify birth, B is the positive value of sigma, T is the max value in V^{anti-transpose}[sigma]
    */
    Real linear_slope(Real B, Real T, Real t, bool death = true) const
    {
        Real slope;
        if (death){
            slope = (t-B)/(T-B);
        }else{ 
            slope = (T-t)/(T-B);
        }
        return slope;
    }


    CriticalSets linear_decrease_death(size_t negative_simplex_idx, Real target_death) const
    {
        Real negative_val = fil_.get_cell_value(negative_simplex_idx);
        if (fil_.cmp(negative_val, target_death)) 
            throw std::runtime_error("Want to decrease death value, but current value < target");

        CriticalSets result;
        auto& v_col = decmp_hom_.v_data[negative_simplex_idx];
        // find min filtration value inside the column
        Real min_val = negative_val;  
        for(auto v_idx_it = v_col.rbegin() ; v_idx_it != v_col.rend() ; ++v_idx_it) {
            auto v_idx = *v_idx_it;
            min_val = std::min(min_val, fil_.get_cell_value(v_idx));    
        }
        // Linear interpolate
        Real slope = linear_slope(min_val, negative_val, target_death, true);
        for(auto v_idx_it = v_col.rbegin() ; v_idx_it != v_col.rend() ; ++v_idx_it) {
            auto v_idx = *v_idx_it;
            Real new_val = min_val + slope * (fil_.get_cell_value(v_idx) - min_val);
            result.emplace_back(new_val, Indices{v_idx});   
        }

        if (result.empty())
            throw std::runtime_error("decrease_death: empty");
        return result;
    }
    
    CriticalSets linear_decrease_deathes((const Indices& indices, const Values& values)) const{
        CriticalSets results;
        for(size_t i = 0 ; i < indices.size() ; ++i) {
            results.emplace_back(linear_decrease_deathes(indices[i], values[i]));
        }
    }

    CriticalSets linear_increase_death(size_t negative_simplex_idx, Real target_death) const
    {
        // input validity check 
        Real negative_val = fil_.get_cell_value(negative_simplex_idx);
        if (fil_.cmp(target_death, negative_val))
            throw std::runtime_error("Want to increase death value, but current value > target");

        const auto& r_cols = decmp_hom_.r_data;
        size_t n_cols = decmp_hom_.v_data.size();
        Int sigma = low(r_cols[negative_simplex_idx]);
        if (not(sigma >= 0 and sigma < r_cols.size()))
            throw std::runtime_error("expected negative simplex");

        // store results in vector of pairs of new target values and indices
        CriticalSets result;
        const auto& u_rows = decmp_hom_.u_data_t;

        for(auto tau_idx: u_rows.at(negative_simplex_idx)) {
            Real tau_val = fil_.get_cell_value(tau_idx);
            if (fil_.cmp(target_death, tau_val)) {
                break;
            }
            
            auto& v_col = decmp_hom_.v_data[tau_idx];
            // find min filtration value inside the column
            // TODO: Make the following into a function
            Real min_val_tau = fil_.get_cell_value(tau_idx);
            for(auto v_idx_it = v_col.rbegin() ; v_idx_it != v_col.rend() ; ++v_idx_it) {
                auto v_idx = *v_idx_it;
                min_val_tau = std::min(min_val_tau, fil_.get_cell_value(v_idx));    
            }
            // Liner interpolate
            // TODO: what is the target value at the anothor peak
            Real slope = linear_slope(min_val_tau, tau_val, target_death-negative_val+tau_val, true);
            for(auto v_idx_it = v_col.rbegin() ; v_idx_it != v_col.rend() ; ++v_idx_it) {
                auto v_idx = *v_idx_it;
                Real new_val = min_val_tau + slope * (fil_.get_cell_value(v_idx) - min_val_tau);
                result.emplace_back(new_val, Indices{v_idx});   
            }
        }

        if (result.empty())
            throw std::runtime_error("increase_death: empty");

        return result;
    }



    CriticalSets linear_decrease_birth(size_t positive_simplex_idx, Real target_birth) const
    {   
        Real positive_val = fil_.get_cell_value(positive_simplex_idx);
        if (fil_.cmp(positive_val, target_birth)) // cur < target
            throw std::runtime_error("target_birth cannot precede current value"); 

        CriticalSets result;
        const auto& u_rows = decmp_coh_.u_data_t;
        for(auto index_in_matrix: u_rows.at(fil_.index_in_matrix(positive_simplex_idx, true))) {
            // get filtration index from the index in anti-transpose matrix  
            auto fil_idx = fil_.index_in_filtration(index_in_matrix, true);
            Real sigma_val = fil_.get_cell_value(fil_idx);
            if (fil_.cmp(sigma_val, target_birth)) {
                break;
            }

            // index in the anti-transpose matrix
            auto& sigma_matrix_col_idx = fil_.index_in_matrix(fil_idx, true);
            auto& v_col = decmp_coh_.v_data.at(sigma_matrix_col_idx); 
            // loop column to find the max value
            Real max_val = sigma_val;
            for(auto index_in_matrix = v_col.rbegin() ; index_in_matrix != v_col.rend() ; ++index_in_matrix) {
                auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
                max_val = std::max(max_val, fil_.get_cell_value(fil_idx));
            }
            // set new value
            Real slope = linear_slope(sigma_val, max_val, target_birth, false);
            for(auto index_in_matrix = v_col.rbegin() ; index_in_matrix != v_col.rend() ; ++index_in_matrix) {
                auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
                Real new_val = target_birth + slope * (fil_.get_cell_value(fil_idx) - sigma_val);
                result.emplace_back(new_val, Indices{fil_idx});   
            }
        }

        if (result.empty())
            throw std::runtime_error("decrease birth : empty");
        return result;
    }
    

    CriticalSets linear_increase_birth(size_t positive_simplex_idx, Real target_birth) const
    {
        Real positive_val = fil_.get_cell_value(positive_simplex_idx);
        if (fil_.cmp(target_birth, positive_val)) // target < cur
            throw std::runtime_error("Want to increase birth value, but target < current value"); 

        CriticalSets result;

        auto& v_col = decmp_coh_.v_data.at(fil_.index_in_matrix(positive_simplex_idx, true));
        // loop column to find the max value
        Real max_val = positive_val;
        for(auto index_in_matrix = v_col.rbegin() ; index_in_matrix != v_col.rend() ; ++index_in_matrix) {
            auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
            max_val = std::max(max_val, fil_.get_cell_value(fil_idx));
        }
        // set new value
        Real slope = linear_slope(positive_val, max_val, target_birth, false);
        for(auto index_in_matrix = v_col.rbegin() ; index_in_matrix != v_col.rend() ; ++index_in_matrix) {
            auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
            Real new_val = target_birth + slope * (fil_.get_cell_value(fil_idx) - positive_val);
            result.emplace_back(new_val, Indices{fil_idx});   
        }

        if (result.empty())
            throw std::runtime_error("increase_death: empty");
        return result;
    }


    Decomposition get_homology_decompostion() const { return decmp_hom_; }
    Decomposition get_cohomology_decompostion() const { return decmp_coh_; }

private:
    // data
    bool negate_;

    Fil fil_;

    Decomposition decmp_hom_;
    Decomposition decmp_coh_;

    Params params_hom_;
    Params params_coh_;

    // methods
    bool cmp(Real a, Real b)
    {
        return negate_ ? a > b : a < b;
    }

    Indices change_birth(size_t positive_simplex_idx, Real target_birth)
    {
        CALI_CXX_MARK_FUNCTION;
        Real current_birth = get_cell_value(positive_simplex_idx);

        if (!decmp_coh_.is_reduced or (params_coh_.compute_u and not decmp_coh_.has_matrix_u())) {
            decmp_coh_.reduce(params_coh_);
        }

        if (cmp(target_birth, current_birth)) // target < curr, decrease
            return decrease_birth(positive_simplex_idx, target_birth);
        else if (fil_.cmp(current_birth, target_birth))
            return increase_birth(positive_simplex_idx, target_birth);
        else
            return {};
    }

    Indices change_death(size_t negative_simplex_idx, Real target_death)
    {
        CALI_CXX_MARK_FUNCTION;
        Real current_death = get_cell_value(negative_simplex_idx);
        if (cmp(target_death, current_death)) // target less than current 
            return decrease_death(negative_simplex_idx, target_death);
        else if (cmp(current_death, target_death))
            return increase_death(negative_simplex_idx, target_death);
        else
            return {};
    }

};

} // namespace
