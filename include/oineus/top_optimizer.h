#pragma once

#include <vector>
#include <utility>

#include "filtration.h"
#include "decomposition.h"
#include "loss.h"

namespace oineus {

struct ComputeFlags {
    bool compute_cohomology {false};
    bool compute_homology_u {false};
    bool compute_cohomology_u {false};
};


template<class Int_, class Real_>
class TopologyOptimizer {
public:
    using Real = Real_;
    using Int = Int_;
    using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;
    using Indices = std::vector<Int>;
    using Values = std::vector<Real>;
    using DgmTarget = std::unordered_map<DgmPoint<size_t>, DgmPoint<Real>>;
    using CriticalSet = std::pair<Real, Indices>;
    using CriticalSets = std::vector<CriticalSet>;

    using Decomposition = VRUDecomposition<Int>;
    using Dgms = Diagrams<Real>;
    using Dgm = typename Dgms::Dgm;
    using Fil = Filtration<Int, Real>;

    struct SimplexTarget {
        Real current_value;
        Real target_value;
        bool is_positive;

        bool increase_birth() const { return is_positive and target_value > current_value; }
        bool decrease_birth() const { return is_positive and target_value < current_value; }
        bool increase_death() const { return not is_positive and target_value > current_value; }
        bool decrease_death() const { return not is_positive and target_value < current_value; }
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
    };

    // initialize lazily, homology only, no reduction
    TopologyOptimizer(const BoundaryMatrix& boundary_matrix, const Values& values, bool negate = false)
            :
            decmp_hom_(boundary_matrix),
            negate_(negate)
    {
    }

    template<class Filtration>
    TopologyOptimizer(const Filtration& fil)
            :
            decmp_hom_(fil, false),
            fil_(fil),
            negate_(fil.negate()) { }

    template<class Filtration>
    TopologyOptimizer(const Filtration& fil, const ComputeFlags& hints)
            :
            decmp_hom_(fil, false),
            fil_(fil),
            negate_(fil.negate())
    {
        if (hints.compute_cohomology) {
            decmp_coh_ = Decomposition(fil, true);
        }

        params_hom_.compute_u = hints.compute_homology_u;
        params_coh_.compute_u = hints.compute_cohomology_u;
    }

    ComputeFlags get_flags(const Target& target)
    {
        bool increase_birth = std::accumulate(target.begin(), target.end(), false, [](bool x, auto kv) { return x or kv.second.increase_birth(); });
        bool decrease_birth = std::accumulate(target.begin(), target.end(), false, [](bool x, auto kv) { return x or kv.second.decrease_birth(); });
        bool increase_death = std::accumulate(target.begin(), target.end(), false, [](bool x, auto kv) { return x or kv.second.increase_death(); });
        bool decrease_death = std::accumulate(target.begin(), target.end(), false, [](bool x, auto kv) { return x or kv.second.decrease_death(); });

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_death;
        result.compute_homology_u = increase_birth;
        result.compute_cohomology_u = decrease_death;

        return result;
    }

    ComputeFlags get_flags(const DgmTarget& target)
    {
        bool increase_birth = false;
        bool increase_death = false;
        bool decrease_birth = false;
        bool decrease_death = false;

        for(const auto& [idx_point, target_point]: target) {

            if (original_values_[idx_point.birth] < target_point.birth) {
                increase_birth = true;
            } else if (original_values_[idx_point.birth] > target_point.birth) {
                decrease_birth = true;
            }

            if (original_values_[idx_point.death] < target_point.death) {
                increase_death = true;
            } else if (original_values_[idx_point.death] > target_point.death) {
                decrease_death = true;
            }
        }

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_death;
        result.compute_homology_u = increase_birth;
        result.compute_cohomology_u = decrease_death;

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
        auto d = get_dimension(index);

        if (!decmp_hom_.is_reduced)
            decmp_hom_.reduce(params_hom_);

        if (decmp_hom_.is_negative(index)) {
            return {value, change_death_x(d, index, value)};
        } else {
            return {value, change_birth_x(d, index, value)};
        }
    }

    CriticalSets singletons(const Indices& indices, const Values& values)
    {
        if (indices.size() != values.size())
            throw std::runtime_error("indices and values must have the same size");

        CriticalSets result;
        result.reserve(indices.size());

        for(size_t i = 0 ; i < indices.size() ; ++i) {
            result.emplace_back(singleton(indices[i], values[i]));
        }

        return result;
    }

    void update(const Values& new_values);

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

    Real get_simplex_value(size_t simplex_idx) const
    {
        return fil_.value_by_sorted_id(simplex_idx);
    }

//    DgmTarget match() const
//    {
//        DgmTarget result;
//        return result;
//    }

    Target dgm_target_to_target(const DgmTarget& dgm_target) const
    {
        Target target;

        for(auto&& [idx_point, target_point]: dgm_target) {
            size_t birth_simplex = idx_point.birth;
            Real current_birth_value = get_simplex_value(birth_simplex);
            Real target_birth_value = target_point.birth;

            if (current_birth_value != target_birth_value)
                target.emplace(birth_simplex, {current_birth_value, target_birth_value, true});

            size_t death_simplex = idx_point.death;
            Real current_death_value = get_simplex_value(death_simplex);
            Real target_death_value = target_point.death;

            if (current_death_value != target_death_value)
                target.emplace(death_simplex, {current_death_value, target_death_value, false});
        }

        return target;
    }

    IndicesValues simplify(Real eps, DenoiseStrategy strategy, dim_type d) const
    {
        IndicesValues result;

        auto index_diagram = decmp_hom_.template index_diagram<Real>(fil_, false, false)[d];

        for(auto p: index_diagram) {
            Real birth = get_simplex_value(p.birth);
            Real death = get_simplex_value(p.death);
            Real pers = abs(death - birth);
            if (pers <= eps) {
                if (strategy == DenoiseStrategy::BirthBirth)
                    result.push_back(p.death, birth);
                else if (strategy == DenoiseStrategy::DeathDeath)
                    result.push_back(p.birth, death);
                else if (strategy == DenoiseStrategy::Midway) {
                    result.push_back(p.birth, (birth + death) / 2);
                    result.push_back(p.death, (birth + death) / 2);
                }
            }
        }
        return result;
    }

    IndicesValues match(typename Diagrams<Real>::Dgm& template_dgm, dim_type d, Real wasserstein_q)
    {
        // set ids in template diagram
        for(hera::id_type i = 0 ; i < template_dgm.size() ; ++i) {
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

        auto current_index_dgm = decmp_hom_.index_diagram(fil_, false, false).get_diagram_in_dimension(d);

        Diagram current_dgm;
        current_dgm.reserve(current_index_dgm.size());

        for(hera::id_type current_dgm_id = 0 ; current_dgm_id < current_index_dgm.size() ; ++current_dgm_id) {

            auto birth_idx = current_index_dgm[current_dgm_id].birth;
            auto death_idx = current_index_dgm[current_dgm_id].death;

            auto birth_val = get_simplex_value(birth_idx);
            auto death_val = get_simplex_value(death_idx);

            // do not include diagonal points
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

            if (template_id >= 0) {
                // matched to off-diagonal point of template diagram

                size_t birth_idx = current_index_dgm.at(current_id).birth;
                size_t death_idx = current_index_dgm.at(current_id).death;

                Real birth_target = template_dgm.at(template_id).birth;
                Real death_target = template_dgm.at(template_id).death;

                result.push_back(birth_idx, birth_target);
                result.push_back(death_idx, death_target);
            }
            // else { }
            // TODO: should we just ignore this point, if it is matched to the diagonal? or simplify it?
        }

        return result;
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
        auto simplex_to_values = convert_critical_sets(critical_sets);
        IndicesValues indvals;

        if (strategy == ConflictStrategy::Max) {
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                Real current_value = get_simplex_value(simplex_idx);
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
            // send critical simplices according to the matching loss
            // average on others
            for(auto&& [simplex_idx, values]: simplex_to_values) {
                // where matching loss tells critical simplices to go
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
    }

    void combine_loss(const Indices& indices, const Values& values, ConflictStrategy strategy)
    {
        return combine_loss(singletons(indices, values), strategy);
    }

    Dgms compute_diagram(bool include_inf_points)
    {
        if (!decmp_hom_.is_reduced)
            decmp_hom_.reduce(params_hom_);

        return decmp_hom_.diagram(fil_, include_inf_points);
    }

private:
    bool negate_;
    bool cmp(Real a, Real b)
    {
        return negate_ ? a > b : a < b;
    }

    Values original_values_;
    Dgm original_diagram_;

    Fil fil_;

    Decomposition decmp_hom_;
    Decomposition decmp_coh_;

    Params params_hom_;
    Params params_coh_;

    Indices change_birth_x(dim_type d, size_t positive_simplex_idx, Real target_birth)
    {
        Real current_birth = original_values_[positive_simplex_idx];
        if (cmp(target_birth, current_birth))
            return decrease_birth_x<Int, Real>(d, positive_simplex_idx, fil_, decmp_coh_, target_birth);
        else if (fil_.cmp(current_birth, target_birth))
            return increase_birth_x<Int, Real>(d, positive_simplex_idx, fil_, decmp_coh_, target_birth);
        else
            return {};
    }

    Indices change_death_x(dim_type d, size_t negative_simplex_idx, Real target_death)
    {
        Real current_death = original_values_[negative_simplex_idx];
        if (cmp(target_death, current_death))
            return decrease_death_x(d, negative_simplex_idx, fil_, decmp_hom_, target_death);
        else if (fil_.cmp(current_death, target_death))
            return increase_death_x(d, negative_simplex_idx, fil_, decmp_hom_, target_death);
        else
            return {};
    }

};
} // namespace
