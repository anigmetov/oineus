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
            negate_(fil.negate()),
            fil_(fil),
            decmp_hom_(fil, false),
            decmp_coh_(fil, true)
    {
        params_hom_.compute_v = true;
        params_coh_.compute_v = true;
        params_hom_.compute_u = true;
        params_coh_.compute_u = true;
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
        params_hom_.clearing_opt = false;
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

        for(const auto& [point, target_point]: target) {

            if (cmp(point.birth, target_point.birth)) {
                increase_birth = true;
            } else if (cmp(target_point.birth, point.birth)) {
                decrease_birth = true;
            }

            if (cmp(point.death, target_point.death)) {
                increase_death = true;
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
        fil_.set_values(new_values);

        decmp_hom_ = Decomposition(fil_, false, n_threads);
        decmp_coh_ = Decomposition(fil_, true, n_threads);
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
        return fil_.get_cell_value(simplex_idx);
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
        for(size_t i = 0 ; i < template_dgm.size() ; ++i) {
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

        for(size_t i = 0 ; i < current_dgm.size() ; ++i) {
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
        assert(fil_.cmp(fil_.get_cell_value(positive_simplex_idx), target_birth));

        Indices result;

        auto& v_col = decmp_coh_.v_data.at(fil_.index_in_matrix(positive_simplex_idx, true));

        for(auto index_in_matrix = v_col.rbegin() ; index_in_matrix != v_col.rend() ; ++index_in_matrix) {
            auto fil_idx = fil_.index_in_filtration(*index_in_matrix, true);
            if (fil_.cmp(target_birth, fil_.get_cell_value(fil_idx)))
                break;

            result.push_back(fil_idx);
        }

        assert(not result.empty());

        return result;
    }

    Indices increase_birth(size_t positive_simplex_idx) const
    {
        return increase_birth(positive_simplex_idx, fil_.infinity());
    }

    Indices decrease_birth(size_t positive_simplex_idx, Real target_birth) const
    {
        assert(fil_.cmp(target_birth, fil_.get_cell_value(positive_simplex_idx)));

        Indices result;

        for(auto index_in_matrix: decmp_coh_.u_data_t.at(fil_.index_in_matrix(positive_simplex_idx, true))) {
            auto fil_idx = fil_.index_in_filtration(index_in_matrix, true);

            if (fil_.cmp(fil_.get_cell_value(fil_idx), target_birth)) {
                break;
            }

            result.push_back(fil_idx);
        }

        assert(not result.empty());

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

        Int sigma = low(r_cols[negative_simplex_idx]);

        assert(sigma >= 0 and sigma < static_cast<Int>(r_cols.size()));

        for(auto tau_idx: u_rows.at(negative_simplex_idx)) {
            if (fil_.cmp(target_death, fil_.get_cell_value(tau_idx))) {
                break;
            }

            if (low(decmp_hom_.r_data[tau_idx]) <= sigma) {
                result.push_back(tau_idx);
            }
        }

        assert(not result.empty());

        return result;
    }

    Indices increase_death(size_t negative_simplex_idx) const
    {
        CALI_CXX_MARK_FUNCTION;
        return increase_death(negative_simplex_idx, fil_.infinity());
    }

    Indices decrease_death(size_t negative_simplex_idx, Real target_death) const
    {
        CALI_CXX_MARK_FUNCTION;
        Indices result;

        auto& r_cols = decmp_hom_.r_data;
        Int sigma = low(r_cols[negative_simplex_idx]);

        assert(sigma >= 0 and sigma < static_cast<Int>(r_cols.size()));

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

        assert(not result.empty());

        return result;
    }

    Indices decrease_death(size_t negative_simplex_idx) const
    {
        CALI_CXX_MARK_FUNCTION;
        return decrease_death(negative_simplex_idx, -fil_.infinity());
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

        if (cmp(target_birth, current_birth))
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
        if (cmp(target_death, current_death))
            return decrease_death(negative_simplex_idx, target_death);
        else if (cmp(current_death, target_death))
            return increase_death(negative_simplex_idx, target_death);
        else
            return {};
    }

};

} // namespace
