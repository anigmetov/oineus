#pragma once

#include <vector>
#include <utility>

#include "decomposition.h"
#include "loss.h"

namespace oineus {

struct ComputeFlags
{
    bool compute_cohomology {false};
    bool compute_homology_u {false};
    bool compute_cohomology {false};
};

//class FiltrationWrapper {
//    std::
//};

template<class Int_, class Real_>
class TopologyOptimizer {
public:
    using Real = Real_;
    using Int = Int_;
    using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;
    using Indices = std::vector<size_t>;
    using Values = std::vector<Real>;
    using Target = std::unordered_map<DgmPoint<size_t>, DgmPoint<Real>>;
    using CriticalSet = std::pair<Real, Indices>;
    using CriticalSets = std::vector<CriticalSet>;
    using IndicesValues = std::pair<Indices, Values>;
    using Decomposition = VRUDecomposition<Int>;

    TopologyOptimizer(const BoundaryMatrix& boundary_matrix, const Values& values)
    {
        // initialize lazily, homology only, no reduction
        dcmp_hom_ = std::make_unique(boundary_matrix);
    }


    TopologyOptimizer(const BoundaryMatrix& boundary_matrix, const Values& values, const ComputeFlags hints);

    template<class Filtration>
    TopologyOptimizer(const Filtration fil);

    template<class Filtration>
    TopologyOptimizer(const Filtration fil, const ComputeFlags& hints)
    {
        dcmp_hom_ = std::make_unique(fil, false);

        if (hints.compute_cohomology) {
            dcmp_coh_ = std::make_unique(fil, true);
        }

        params_hom_.compute_u = hints.compute_homology_u;
        params_coh_.compute_u = hints.compute_cohomology_u;
    }

    ComputeFlags get_flags(const Target& target)
    {
        bool increase_birth = false;
        bool increase_death = false;
        bool decrease_birth = false;
        bool decrease_death = false;

        for(const auto& [idx_point, target_point] : target) {

            if (original_values[idx_point.birth] < target_point.birth)
                increase_birth = true;

            if (original_values[idx_point.birth] > target_point.birth)
                decrease_birth = true;

            if (original_values[idx_point.death] < target_point.death)
                increase_death = true;

            if (original_values[idx_point.death] > target_point.death)
                decrease_death = true;
        }

        ComputeFlags result;

        result.compute_cohomology = decrease_birth or increase_death;
        result.compute_homology_u = increase_birth;
        result.compute_cohomology_u = decrease_death;

        return result;
    }

    CriticalSet singleton(size_t index, Real value)
    {
        if (!dcmp_hom_->is_reduced)
            dcmp_hom_->reduce(params_hom_);

        if (dcmp_hom_->is_negative(index)) {
        } else {
        }
    }

    CriticalSets singletons(const Indices& indices, const Values& values)
    {

    }

    void update(new_values);

    void combine_loss(const CriticalSets& critical_sets, ConflictStrategy strategy);

    void combine_loss(const Indices& indices, const Values& values, ConflictStrategy strategy)
    {
        return combine_loss(singletons(indices, values), strategy);
    }

private:
    bool negate_;
    bool cmp(Real a, Real b);

    Values original_values_;
    Diagram original_diagram_;

    std::unique_ptr<Decomposition> dcmp_hom_ {nullptr};
    std::unique_ptr<Decomposition> dcmp_coh_ {nullptr};

    Params params_hom_;
    Params params_coh_;


    Indices change_birth_x(dim_type d, size_t positive_simplex_idx, Real target_birth)
    {
        Real current_birth = original_values_[positive_simplex_idx];
        if (cmp(target_birth, current_birth))
            return decrease_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
        else if (fil.cmp(current_birth, target_birth))
            return increase_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
        else
            return {};
    }


};

    template<class Int, class Real, class L>
    std::vector<Int> increase_birth_x(dim_type d, size_t positive_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_birth)
    {
        if (not decmp.dualize())
            throw std::runtime_error("increase_birth_x: expected cohomology");

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

        for(auto index_in_matrix: decmp.u_data_t.at(fil.index_in_matrix(positive_simplex_idx, decmp.dualize()))) {
            auto fil_idx = fil.index_in_filtration(index_in_matrix, decmp.dualize());
            const auto& sigma = fil.simplices()[fil_idx];

            if (fil.cmp(sigma.value(), target_birth)) {
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
        if (decmp.dualize())
            throw std::runtime_error("increase_death_x: expected homology, got cohomology");

        std::vector<Int> result;

        const auto& u_rows = decmp.u_data_t;
        const auto& r_cols = decmp.r_data;
        const auto& simplices = fil.simplices();

        size_t n_cols = decmp.v_data.size();
        Int sigma = low(r_cols[negative_simplex_idx]);

        if (not(sigma >= 0 and sigma < r_cols.size()))
            throw std::runtime_error("expected negative simplex");

        for(auto tau_idx: u_rows.at(negative_simplex_idx)) {
            const auto& tau = simplices.at(tau_idx);
            assert(tau.dim() == d);
            if (fil.cmp(target_death, tau.value())) {
                break;
            }

            if (low(decmp.r_data[tau_idx]) <= sigma)
                result.push_back(tau_idx);
        }

        if (result.empty())
            throw std::runtime_error("increase_death_x: empty");

        return result;
    }

    template<class Int, class Real, class L>
    std::vector<Int> decrease_death_x(dim_type d, size_t negative_simplex_idx, const oineus::Filtration<Int, Real, L>& fil, const oineus::VRUDecomposition<Int>& decmp, Real target_death)
    {
        if (decmp.dualize())
            throw std::runtime_error("decrease_death_x: expected homology, got cohomology");

        std::vector<Int> result;

        auto& r_cols = decmp.r_data;
        Int sigma = low(r_cols[negative_simplex_idx]);

        assert(sigma >= 0 and sigma < r_cols.size());

        auto& v_col = decmp.v_data[negative_simplex_idx];

        for(auto tau_idx_it = v_col.rbegin(); tau_idx_it != v_col.rend(); ++tau_idx_it) {
            auto tau_idx = *tau_idx_it;
            const auto& tau = fil.simplices()[tau_idx];
            assert(tau.dim() == d + 1);

            if (fil.cmp(tau.value(), target_death))
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

    template<class int, class real, class l>
    std::vector<int> change_birth_x(dim_type d, size_t positive_simplex_idx, const oineus::filtration<int, real, l>& fil, const oineus::vrudecomposition<int>& decmp_coh, real target_birth)
    {
        real current_birth = fil.simplices()[positive_simplex_idx].value();
        if (fil.cmp(target_birth, current_birth))
            return decrease_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
        else if (fil.cmp(current_birth, target_birth))
            return increase_birth_x(d, positive_simplex_idx, fil, decmp_coh, target_birth);
        else
            return {};
    }



} // namespace
