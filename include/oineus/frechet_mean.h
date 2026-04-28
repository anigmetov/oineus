#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagram.h"
#include "hera/wasserstein.h"
#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

namespace oineus {

    enum class DiagramPlaneDomain {
        AboveDiagonal,
        BelowDiagonal,
        Mixed
    };

    enum class FrechetMeanInit {
        Custom,
        FirstDiagram,
        MedoidDiagram,
        RandomDiagram,
        Grid,
    };

    struct FrechetMeanInitGridParams {
        size_t n_x_bins {16};
        size_t n_y_bins {16};
        DiagramPlaneDomain domain {DiagramPlaneDomain::AboveDiagonal};
    };

    template<class Real>
    struct FrechetMeanInitRandomParams {
        Real noise_scale {1.0};
        size_t random_seed {42};
        DiagramPlaneDomain domain {DiagramPlaneDomain::AboveDiagonal};
    };

    template<class Real>
    struct FrechetMeanParams {
        size_t max_iter {100};
        Real tol {static_cast<Real>(1e-7)};
        Real wasserstein_delta {static_cast<Real>(1e-2)};
        Real internal_p {std::numeric_limits<Real>::infinity()};
        FrechetMeanInit init_strategy {FrechetMeanInit::Grid};
        DiagramPlaneDomain domain {DiagramPlaneDomain::AboveDiagonal};
        bool ignore_infinite_points {false};
        int n_threads {1};
        FrechetMeanInitRandomParams<Real> random_init_params {};
        FrechetMeanInitGridParams grid_init_params {};
    };

    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_first_diagram(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams);

    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_random_diagram(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const FrechetMeanInitRandomParams<Real>& params = {});

    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_medoid_diagram(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights = {},
            int n_threads = 1);

    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_diagonal_grid(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights = {},
            const FrechetMeanInitGridParams& params = {});

    template<class Real>
    typename Diagrams<Real>::Dgm frechet_mean(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights = {},
            const FrechetMeanParams<Real>& params = {},
            const typename Diagrams<Real>::Dgm& custom_initial_barycenter = {});

    // Split a diagram into finite points and the four supported one-infinite-coordinate families.
    template<class Real>
    struct SplitDiagram {
        using Dgm = typename Diagrams<Real>::Dgm;

        Dgm finite;
        Dgm neg_inf_birth;
        Dgm pos_inf_birth;
        Dgm neg_inf_death;
        Dgm pos_inf_death;
    };

    // Hera matching expects dense ids in [0, n).
    template<class Real>
    inline void assign_sequential_ids(typename Diagrams<Real>::Dgm& diagram)
    {
        for(size_t i = 0; i < diagram.size(); ++i)
            diagram[i].id = static_cast<id_type>(i);
    }

    // Build a trivial matching when the diagrams are already equal.
    template<class Real>
    inline hera::AuctionResult<Real> zero_cost_matching_result(
            typename Diagrams<Real>::Dgm left,
            typename Diagrams<Real>::Dgm right)
    {
        auto point_less = [](const auto& a, const auto& b) {
            return std::tie(a.birth, a.death, a.id) < std::tie(b.birth, b.death, b.id);
        };

        std::sort(left.begin(), left.end(), point_less);
        std::sort(right.begin(), right.end(), point_less);

        hera::AuctionResult<Real> result;
        const size_t n = std::min(left.size(), right.size());
        for (size_t i = 0; i < n; ++i)
            result.add_to_matching(static_cast<int>(left[i].id), static_cast<int>(right[i].id));
        result.distance = static_cast<Real>(0);
        result.cost = static_cast<Real>(0);
        return result;
    }

    // Compute the q=2 Wasserstein matching for the finite parts only.
    template<class Real>
    inline hera::AuctionResult<Real> wasserstein_matching_result_q2(
            typename Diagrams<Real>::Dgm barycenter,
            typename Diagrams<Real>::Dgm diagram,
            Real delta,
            Real internal_p)
    {
        assign_sequential_ids<Real>(barycenter);
        assign_sequential_ids<Real>(diagram);

        if (hera::ws::are_equal(barycenter, diagram))
            return zero_cost_matching_result<Real>(std::move(barycenter), std::move(diagram));

        hera::AuctionParams<Real> params;
        params.return_matching = true;
        params.match_inf_points = true;
        params.wasserstein_power = static_cast<Real>(2);
        params.delta = delta;
        params.internal_p = std::isinf(internal_p) ? hera::get_infinity<Real>() : internal_p;

        return hera::wasserstein_cost_detailed(barycenter, diagram, params);
    }

    // Map a one-infinite-coordinate point to one of the four supported families.
    template<class Real>
    inline int infinite_family(const typename Diagrams<Real>::Point& point)
    {
        if (!point.is_inf())
            return 0;

        const bool birth_pos = DgmPoint<Real>::is_plus_inf(point.birth);
        const bool birth_neg = DgmPoint<Real>::is_minus_inf(point.birth);
        const bool death_pos = DgmPoint<Real>::is_plus_inf(point.death);
        const bool death_neg = DgmPoint<Real>::is_minus_inf(point.death);
        const bool finite_birth = !birth_pos && !birth_neg;
        const bool finite_death = !death_pos && !death_neg;

        if (birth_neg && finite_death)
            return 1;
        if (birth_pos && finite_death)
            return 2;
        if (finite_birth && death_neg)
            return 3;
        if (finite_birth && death_pos)
            return 4;

        throw std::runtime_error("frechet_mean: unsupported infinite point; expected exactly one infinite coordinate");
    }

    // Reconstruct an infinite point from its family id and finite coordinate.
    template<class Real>
    inline typename Diagrams<Real>::Point make_infinite_point_from_family(
            int family,
            Real finite_value)
    {
        switch (family) {
        case 1: return {DgmPoint<Real>::minus_inf(), finite_value};
        case 2: return {DgmPoint<Real>::plus_inf(), finite_value};
        case 3: return {finite_value, DgmPoint<Real>::minus_inf()};
        case 4: return {finite_value, DgmPoint<Real>::plus_inf()};
        default:
            throw std::runtime_error("frechet_mean: unknown infinite family");
        }
    }

    // Extract the finite coordinate used by a given infinite family.
    template<class Real>
    inline Real finite_coordinate_for_family(
            const typename Diagrams<Real>::Point& point,
            int family)
    {
        switch (family) {
        case 1:
        case 2:
            return point.death;
        case 3:
        case 4:
            return point.birth;
        default:
            throw std::runtime_error("frechet_mean: unknown infinite family");
        }
    }

    // Return the container that stores one of the four infinite families.
    template<class Real>
    inline typename SplitDiagram<Real>::Dgm& infinite_family_points(
            SplitDiagram<Real>& split_diagram,
            int family)
    {
        switch (family) {
        case 1: return split_diagram.neg_inf_birth;
        case 2: return split_diagram.pos_inf_birth;
        case 3: return split_diagram.neg_inf_death;
        case 4: return split_diagram.pos_inf_death;
        default:
            throw std::runtime_error("frechet_mean: unknown infinite family");
        }
    }

    template<class Real>
    inline const typename SplitDiagram<Real>::Dgm& infinite_family_points(
            const SplitDiagram<Real>& split_diagram,
            int family)
    {
        switch (family) {
        case 1: return split_diagram.neg_inf_birth;
        case 2: return split_diagram.pos_inf_birth;
        case 3: return split_diagram.neg_inf_death;
        case 4: return split_diagram.pos_inf_death;
        default:
            throw std::runtime_error("frechet_mean: unknown infinite family");
        }
    }

    // Sort an infinite family by its finite coordinate so family-wise matching is positional.
    template<class Real>
    inline void sort_infinite_family(
            typename SplitDiagram<Real>::Dgm& family_points,
            int family)
    {
        std::sort(family_points.begin(), family_points.end(), [family](const auto& lhs, const auto& rhs) {
            return finite_coordinate_for_family<Real>(lhs, family) < finite_coordinate_for_family<Real>(rhs, family);
        });
    }

    // Separate finite points from essential families once up front.
    template<class Real>
    inline SplitDiagram<Real> split_diagram(
            const typename Diagrams<Real>::Dgm& diagram)
    {
        SplitDiagram<Real> result;
        result.finite.reserve(diagram.size());

        for (const auto& point : diagram) {
            if (!point.is_inf()) {
                result.finite.push_back(point);
                continue;
            }

            infinite_family_points<Real>(result, infinite_family<Real>(point)).push_back(point);
        }

        for (int family = 1; family <= 4; ++family)
            sort_infinite_family<Real>(infinite_family_points<Real>(result, family), family);

        return result;
    }

    // All diagrams must have the same essential-family cardinalities.
    template<class Real>
    inline void validate_same_infinite_cardinalities(
            const std::vector<SplitDiagram<Real>>& diagrams)
    {
        if (diagrams.empty())
            return;

        for (int family = 1; family <= 4; ++family) {
            const size_t expected = infinite_family_points<Real>(diagrams.front(), family).size();
            for (size_t diagram_idx = 1; diagram_idx < diagrams.size(); ++diagram_idx) {
                const size_t actual = infinite_family_points<Real>(diagrams[diagram_idx], family).size();
                if (actual != expected)
                    throw std::runtime_error("frechet_mean: input diagrams have incompatible essential-point cardinalities");
            }
        }
    }

    // The current barycenter must be compatible with every input diagram on essential families.
    template<class Real>
    inline void validate_matching_infinite_cardinalities(
            const SplitDiagram<Real>& lhs,
            const SplitDiagram<Real>& rhs,
            const std::string& what)
    {
        for (int family = 1; family <= 4; ++family) {
            if (infinite_family_points<Real>(lhs, family).size() != infinite_family_points<Real>(rhs, family).size())
                throw std::runtime_error(what);
        }
    }

    // Normalize user-supplied diagram weights and default to the uniform distribution.
    template<class Real>
    inline std::vector<Real> normalized_frechet_weights(
            size_t n_diagrams,
            const std::vector<Real>& weights)
    {
        if (n_diagrams == 0)
            return {};

        if (weights.empty())
            return std::vector<Real>(n_diagrams, static_cast<Real>(1) / static_cast<Real>(n_diagrams));

        if (weights.size() != n_diagrams)
            throw std::runtime_error("frechet_mean: weights must have the same length as diagrams");

        Real weight_sum = static_cast<Real>(0);
        for (const auto weight : weights) {
            if (weight < static_cast<Real>(0))
                throw std::runtime_error("frechet_mean: weights must be nonnegative");
            weight_sum += weight;
        }

        if (weight_sum <= static_cast<Real>(0))
            throw std::runtime_error("frechet_mean: weights must sum to a positive value");

        std::vector<Real> normalized = weights;
        for (auto& weight : normalized)
            weight /= weight_sum;

        return normalized;
    }

    // Reassemble a split diagram back into the usual vector-of-points representation.
    template<class Real>
    inline typename Diagrams<Real>::Dgm combine_split_diagram(
            SplitDiagram<Real>&& split_diagram)
    {
        typename Diagrams<Real>::Dgm result = std::move(split_diagram.finite);
        result.reserve(result.size()
                + split_diagram.neg_inf_birth.size()
                + split_diagram.pos_inf_birth.size()
                + split_diagram.neg_inf_death.size()
                + split_diagram.pos_inf_death.size());

        result.insert(result.end(),
                std::make_move_iterator(split_diagram.neg_inf_birth.begin()),
                std::make_move_iterator(split_diagram.neg_inf_birth.end()));
        result.insert(result.end(),
                std::make_move_iterator(split_diagram.pos_inf_birth.begin()),
                std::make_move_iterator(split_diagram.pos_inf_birth.end()));
        result.insert(result.end(),
                std::make_move_iterator(split_diagram.neg_inf_death.begin()),
                std::make_move_iterator(split_diagram.neg_inf_death.end()));
        result.insert(result.end(),
                std::make_move_iterator(split_diagram.pos_inf_death.begin()),
                std::make_move_iterator(split_diagram.pos_inf_death.end()));

        return result;
    }

    // Evaluate the q=2 diagram cost by combining Hera on finite points with 1D costs on infinite families.
    template<class Real>
    inline Real wasserstein_cost_q2(
            const SplitDiagram<Real>& lhs,
            const SplitDiagram<Real>& rhs,
            Real delta,
            Real internal_p)
    {
        validate_matching_infinite_cardinalities<Real>(
                lhs, rhs,
                "frechet_mean: input diagrams have incompatible essential-point cardinalities");

        Real total_cost = wasserstein_matching_result_q2<Real>(lhs.finite, rhs.finite, delta, internal_p).cost;

        for (int family = 1; family <= 4; ++family) {
            const auto& lhs_family = infinite_family_points<Real>(lhs, family);
            const auto& rhs_family = infinite_family_points<Real>(rhs, family);
            for (size_t i = 0; i < lhs_family.size(); ++i) {
                const Real diff = finite_coordinate_for_family<Real>(lhs_family[i], family)
                        - finite_coordinate_for_family<Real>(rhs_family[i], family);
                total_cost += diff * diff;
            }
        }

        return total_cost;
    }

    // Exact q=2 Wasserstein barycenter on the real line: sort each diagram and average coordinates with weights.
    template<class Real>
    inline std::vector<Real> wasserstein_barycenter_1d(
            std::vector<std::vector<Real>> coordinates,
            const std::vector<Real>& weights)
    {
        if (coordinates.empty())
            return {};

        const size_t family_size = coordinates.front().size();
        for (auto& coords : coordinates) {
            if (coords.size() != family_size)
                throw std::runtime_error("frechet_mean: input diagrams have incompatible essential-point cardinalities");
            std::sort(coords.begin(), coords.end());
        }

        std::vector<Real> barycenter(family_size, static_cast<Real>(0));
        for (size_t diagram_idx = 0; diagram_idx < coordinates.size(); ++diagram_idx)
            for (size_t point_idx = 0; point_idx < family_size; ++point_idx)
                barycenter[point_idx] += weights[diagram_idx] * coordinates[diagram_idx][point_idx];

        return barycenter;
    }

    // Closed-form q=2 update for one finite barycenter point, including weighted matches to the diagonal.
    template<class Real>
    inline typename Diagrams<Real>::Point update_finite_barycenter_point_q2(
            const std::vector<typename Diagrams<Real>::Point>& off_diagonal_points,
            const std::vector<Real>& off_diagonal_weights,
            Real diagonal_weight,
            Real collapse_tol)
    {
        using Point = typename Diagrams<Real>::Point;

        if (off_diagonal_points.empty())
            return {static_cast<Real>(0), static_cast<Real>(0)};

        const Real w = std::accumulate(off_diagonal_weights.begin(), off_diagonal_weights.end(), static_cast<Real>(0));
        const Real d = diagonal_weight;

        Real sum_birth = static_cast<Real>(0);
        Real sum_death = static_cast<Real>(0);
        for (size_t i = 0; i < off_diagonal_points.size(); ++i) {
            sum_birth += off_diagonal_weights[i] * off_diagonal_points[i].birth;
            sum_death += off_diagonal_weights[i] * off_diagonal_points[i].death;
        }

        Point result;

        if (d == static_cast<Real>(0)) {
            result.birth = sum_birth / w;
            result.death = sum_death / w;
        } else {
            const Real a = w + d / static_cast<Real>(2);
            const Real b = d / static_cast<Real>(2);
            const Real det = w * (w + d);
            result.birth = (a * sum_birth + b * sum_death) / det;
            result.death = (b * sum_birth + a * sum_death) / det;
        }

        if (std::abs(result.death - result.birth) <= collapse_tol)
            return {result.birth, result.birth};

        return result;
    }

    // Solve the exact weighted 1D barycenter problem for one infinite family.
    template<class Real>
    inline typename SplitDiagram<Real>::Dgm update_infinite_family_q2(
            const std::vector<SplitDiagram<Real>>& diagrams,
            const std::vector<Real>& weights,
            int family)
    {
        if (diagrams.empty())
            return {};

        const size_t family_size = infinite_family_points<Real>(diagrams.front(), family).size();
        std::vector<std::vector<Real>> coordinates;
        coordinates.reserve(diagrams.size());

        for (const auto& diagram : diagrams) {
            const auto& family_points = infinite_family_points<Real>(diagram, family);
            std::vector<Real> coords;
            coords.reserve(family_points.size());
            for (const auto& point : family_points)
                coords.push_back(finite_coordinate_for_family<Real>(point, family));
            coordinates.push_back(std::move(coords));
        }

        auto barycenter_coords = wasserstein_barycenter_1d<Real>(std::move(coordinates), weights);

        typename SplitDiagram<Real>::Dgm result;
        result.reserve(family_size);
        for (const auto coord : barycenter_coords)
            result.push_back(make_infinite_point_from_family<Real>(family, coord));

        return result;
    }

    // Measure the largest coordinate-wise change between two aligned points.
    template<class Real>
    inline Real convergence_distance(
            const typename Diagrams<Real>::Point& lhs,
            const typename Diagrams<Real>::Point& rhs)
    {
        if (lhs.is_inf() || rhs.is_inf()) {
            if (infinite_family<Real>(lhs) != infinite_family<Real>(rhs))
                return std::numeric_limits<Real>::infinity();

            switch (infinite_family<Real>(lhs)) {
            case 1:
            case 2:
                return std::abs(lhs.death - rhs.death);
            case 3:
            case 4:
                return std::abs(lhs.birth - rhs.birth);
            default:
                return static_cast<Real>(0);
            }
        }

        return std::max(std::abs(lhs.birth - rhs.birth), std::abs(lhs.death - rhs.death));
    }

    // Choose the requested initialization strategy.
    template<class Real>
    inline typename Diagrams<Real>::Dgm initialize_frechet_mean_from_params(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights,
            const FrechetMeanParams<Real>& params,
            const typename Diagrams<Real>::Dgm& custom_initial_barycenter)
    {
        switch (params.init_strategy) {
        case FrechetMeanInit::Custom:
            if (custom_initial_barycenter.empty() && !diagrams.empty())
                throw std::runtime_error("frechet_mean: custom init strategy requires a non-empty initial barycenter");
            return custom_initial_barycenter;
        case FrechetMeanInit::FirstDiagram:
            return init_frechet_mean_first_diagram<Real>(diagrams);
        case FrechetMeanInit::MedoidDiagram:
            return init_frechet_mean_medoid_diagram<Real>(diagrams, weights, params.n_threads);
        case FrechetMeanInit::RandomDiagram: {
            auto random_params = params.random_init_params;
            random_params.domain = params.domain;
            return init_frechet_mean_random_diagram<Real>(diagrams, random_params);
        }
        case FrechetMeanInit::Grid: {
            auto grid_params = params.grid_init_params;
            grid_params.domain = params.domain;
            return init_frechet_mean_diagonal_grid<Real>(diagrams, weights, grid_params);
        }
        default:
            throw std::runtime_error("frechet_mean: unsupported initialization strategy");
        }
    }

    // Pick the input diagram minimizing the sum of q=2 Wasserstein costs to all others.
    template<class Real>
    inline typename Diagrams<Real>::Dgm medoid_diagram_with_params(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights,
            Real delta,
            Real internal_p,
            int n_threads = 1)
    {
        if (diagrams.empty())
            return {};

        std::vector<SplitDiagram<Real>> split_diagrams;
        split_diagrams.reserve(diagrams.size());
        for (const auto& diagram : diagrams)
            split_diagrams.push_back(split_diagram<Real>(diagram));

        validate_same_infinite_cardinalities<Real>(split_diagrams);

        std::vector<Real> per_i_cost(diagrams.size(), std::numeric_limits<Real>::infinity());

        auto compute_cost_for_i = [&](size_t i) {
            Real total_cost = static_cast<Real>(0);
            for (size_t j = 0; j < diagrams.size(); ++j) {
                if (i == j)
                    continue;
                total_cost += weights[j] * wasserstein_cost_q2<Real>(split_diagrams[i], split_diagrams[j], delta, internal_p);
            }
            per_i_cost[i] = total_cost;
        };

        if (n_threads <= 1) {
            for (size_t i = 0; i < diagrams.size(); ++i)
                compute_cost_for_i(i);
        } else {
            tf::Executor executor(n_threads);
            tf::Taskflow flow;
            flow.for_each_index(size_t(0), diagrams.size(), size_t(1), compute_cost_for_i);
            executor.run(flow).get();
        }

        const auto best_it = std::min_element(per_i_cost.begin(), per_i_cost.end());
        const size_t best_idx = static_cast<size_t>(std::distance(per_i_cost.begin(), best_it));
        return diagrams[best_idx];
    }

    // Initialize from the first diagram verbatim.
    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_first_diagram(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams)
    {
        if (diagrams.empty())
            return {};
        return diagrams.front();
    }

    // Initialize from a random diagram and perturb it without changing the half-plane side.
    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_random_diagram(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const FrechetMeanInitRandomParams<Real>& params)
    {
        if (diagrams.empty())
            return {};

        assert(params.domain != DiagramPlaneDomain::AboveDiagonal
                || std::all_of(diagrams.begin(), diagrams.end(), [](const Diagram& diagram) {
                    return std::all_of(diagram.begin(), diagram.end(), [](const auto& point) { return point.birth <= point.death; });
                }));
        assert(params.domain != DiagramPlaneDomain::BelowDiagonal
                || std::all_of(diagrams.begin(), diagrams.end(), [](const Diagram& diagram) {
                    return std::all_of(diagram.begin(), diagram.end(), [](const auto& point) { return point.birth >= point.death; });
                }));

        std::mt19937 gen(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_int_distribution<size_t> diagram_dist(0, diagrams.size() - 1);
        std::normal_distribution<Real> noise_dist(static_cast<Real>(0), params.noise_scale);

        auto result = diagrams[diagram_dist(gen)];

        for (auto& point : result) {
            if (!point.is_inf()) {
                const Real shift = noise_dist(gen);
                point.birth += shift;
                point.death += shift;
            } else {
                const int family = infinite_family<Real>(point);
                switch (family) {
                case 1:
                case 2:
                    point.death += noise_dist(gen);
                    break;
                case 3:
                case 4:
                    point.birth += noise_dist(gen);
                    break;
                default:
                    break;
                }
            }
        }

        return result;
    }

    // Initialize from the diagram with minimum total pairwise q=2 Wasserstein cost.
    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_medoid_diagram(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights,
            int n_threads)
    {
        const auto normalized_weights = normalized_frechet_weights<Real>(diagrams.size(), weights);
        return medoid_diagram_with_params<Real>(
                diagrams,
                normalized_weights,
                static_cast<Real>(1e-2),
                std::numeric_limits<Real>::infinity(),
                n_threads);
    }

    // Initialize by rotating points by 45 degrees, binning them, and averaging each occupied cell.
    template<class Real>
    typename Diagrams<Real>::Dgm init_frechet_mean_diagonal_grid(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights,
            const FrechetMeanInitGridParams& params)
    {
        using Point = typename Diagrams<Real>::Point;

        assert(params.domain != DiagramPlaneDomain::AboveDiagonal
                || std::all_of(diagrams.begin(), diagrams.end(), [](const Diagram& diagram) {
                    return std::all_of(diagram.begin(), diagram.end(), [](const auto& point) { return point.birth <= point.death; });
                }));
        assert(params.domain != DiagramPlaneDomain::BelowDiagonal
                || std::all_of(diagrams.begin(), diagrams.end(), [](const Diagram& diagram) {
                    return std::all_of(diagram.begin(), diagram.end(), [](const auto& point) { return point.birth >= point.death; });
                }));

        const auto normalized_weights = normalized_frechet_weights<Real>(diagrams.size(), weights);
        typename Diagrams<Real>::Dgm result;

        struct RotatedPoint {
            Real x;
            Real y;
            Real weight;
            Point point;
        };

        std::vector<RotatedPoint> finite_points;
        std::vector<SplitDiagram<Real>> split_diagrams;
        split_diagrams.reserve(diagrams.size());

        constexpr Real inv_sqrt2 = static_cast<Real>(0.70710678118654752440);

        for (size_t diagram_idx = 0; diagram_idx < diagrams.size(); ++diagram_idx) {
            const auto& diagram = diagrams[diagram_idx];
            auto split = split_diagram<Real>(diagram);
            for (const auto& point : split.finite) {
                const Real rx = (point.birth + point.death) * inv_sqrt2;
                const Real ry = (point.death - point.birth) * inv_sqrt2;
                finite_points.push_back({rx, ry, normalized_weights[diagram_idx], point});
            }
            split_diagrams.push_back(std::move(split));
        }

        validate_same_infinite_cardinalities<Real>(split_diagrams);

        if (!finite_points.empty()) {
            Real min_x = finite_points.front().x;
            Real max_x = finite_points.front().x;
            Real min_y = finite_points.front().y;
            Real max_y = finite_points.front().y;

            for (const auto& point : finite_points) {
                min_x = std::min(min_x, point.x);
                max_x = std::max(max_x, point.x);
                min_y = std::min(min_y, point.y);
                max_y = std::max(max_y, point.y);
            }

            const Real x_span = std::max(max_x - min_x, static_cast<Real>(1e-12));
            const Real y_span = std::max(max_y - min_y, static_cast<Real>(1e-12));

            std::vector<std::vector<RotatedPoint>> bins(params.n_x_bins * params.n_y_bins);

            for (const auto& point : finite_points) {
                const Real x_rel = (point.x - min_x) / x_span;
                const Real y_rel = (point.y - min_y) / y_span;
                const size_t x_idx = std::min(params.n_x_bins - 1, static_cast<size_t>(std::floor(x_rel * params.n_x_bins)));
                const size_t y_idx = std::min(params.n_y_bins - 1, static_cast<size_t>(std::floor(y_rel * params.n_y_bins)));
                bins[x_idx * params.n_y_bins + y_idx].push_back(point);
            }

            for (const auto& cell : bins) {
                if (cell.empty())
                    continue;

                Real mean_birth = static_cast<Real>(0);
                Real mean_death = static_cast<Real>(0);
                Real total_weight = static_cast<Real>(0);
                for (const auto& point : cell) {
                    mean_birth += point.weight * point.point.birth;
                    mean_death += point.weight * point.point.death;
                    total_weight += point.weight;
                }
                mean_birth /= total_weight;
                mean_death /= total_weight;
                result.emplace_back(mean_birth, mean_death);
            }
        }

        for (int family = 1; family <= 4; ++family) {
            auto family_points = update_infinite_family_q2<Real>(split_diagrams, normalized_weights, family);
            result.insert(result.end(), family_points.begin(), family_points.end());
        }

        return result;
    }

    // Lloyd-style local descent for the q=2 Fréchet mean of persistence diagrams.
    template<class Real>
    typename Diagrams<Real>::Dgm frechet_mean(
            const std::vector<typename Diagrams<Real>::Dgm>& diagrams,
            const std::vector<Real>& weights,
            const FrechetMeanParams<Real>& params,
            const typename Diagrams<Real>::Dgm& custom_initial_barycenter)
    {
        using Diagram = typename Diagrams<Real>::Dgm;
        using Point = typename Diagrams<Real>::Point;

        if (diagrams.empty())
            return {};

        if (params.wasserstein_delta <= static_cast<Real>(0))
            throw std::runtime_error("frechet_mean: wasserstein_delta must be positive");

        assert(params.domain != DiagramPlaneDomain::AboveDiagonal
                || std::all_of(diagrams.begin(), diagrams.end(), [](const Diagram& diagram) {
                    return std::all_of(diagram.begin(), diagram.end(), [](const auto& point) { return point.birth < point.death; });
                }));
        assert(params.domain != DiagramPlaneDomain::BelowDiagonal
                || std::all_of(diagrams.begin(), diagrams.end(), [](const Diagram& diagram) {
                    return std::all_of(diagram.begin(), diagram.end(), [](const auto& point) { return point.birth > point.death; });
                }));

        const auto normalized_weights = normalized_frechet_weights<Real>(diagrams.size(), weights);
        std::vector<SplitDiagram<Real>> split_diagrams;
        split_diagrams.reserve(diagrams.size());
        for (const auto& diagram : diagrams) {
            split_diagrams.push_back(split_diagram<Real>(diagram));
        }

        if (params.ignore_infinite_points) {
            for (auto& split : split_diagrams) {
                split.neg_inf_birth.clear();
                split.pos_inf_birth.clear();
                split.neg_inf_death.clear();
                split.pos_inf_death.clear();
            }
        }

        validate_same_infinite_cardinalities<Real>(split_diagrams);

        std::vector<Diagram> diagrams_for_init;
        const std::vector<Diagram>* init_diagrams = &diagrams;

        if (params.ignore_infinite_points) {
            diagrams_for_init.reserve(split_diagrams.size());
            for (const auto& split : split_diagrams)
                diagrams_for_init.push_back(split.finite);
            init_diagrams = &diagrams_for_init;
        }

        Diagram barycenter = initialize_frechet_mean_from_params<Real>(
                *init_diagrams,
                normalized_weights,
                params,
                custom_initial_barycenter);

        assert(params.domain != DiagramPlaneDomain::AboveDiagonal
                || std::all_of(barycenter.begin(), barycenter.end(), [](const auto& point) { return point.birth <= point.death; }));
        assert(params.domain != DiagramPlaneDomain::BelowDiagonal
                || std::all_of(barycenter.begin(), barycenter.end(), [](const auto& point) { return point.birth >= point.death; }));

        if (barycenter.empty())
            return {};

        SplitDiagram<Real> barycenter_parts = split_diagram<Real>(barycenter);
        if (params.ignore_infinite_points) {
            barycenter_parts.neg_inf_birth.clear();
            barycenter_parts.pos_inf_birth.clear();
            barycenter_parts.neg_inf_death.clear();
            barycenter_parts.pos_inf_death.clear();
        }

        validate_matching_infinite_cardinalities<Real>(
                barycenter_parts,
                split_diagrams.front(),
                "frechet_mean: initial barycenter has incompatible essential-point cardinalities");

        SplitDiagram<Real> infinite_barycenter;
        if (!params.ignore_infinite_points) {
            infinite_barycenter.neg_inf_birth = update_infinite_family_q2<Real>(split_diagrams, normalized_weights, 1);
            infinite_barycenter.pos_inf_birth = update_infinite_family_q2<Real>(split_diagrams, normalized_weights, 2);
            infinite_barycenter.neg_inf_death = update_infinite_family_q2<Real>(split_diagrams, normalized_weights, 3);
            infinite_barycenter.pos_inf_death = update_infinite_family_q2<Real>(split_diagrams, normalized_weights, 4);
        }

        for (size_t iter = 0; iter < params.max_iter; ++iter) {
            std::vector<hera::AuctionResult<Real>> matchings(split_diagrams.size());

            auto compute_matching = [&](size_t i) {
                auto matching = wasserstein_matching_result_q2<Real>(
                        barycenter_parts.finite, split_diagrams[i].finite,
                        params.wasserstein_delta, params.internal_p);
                if (std::isinf(matching.cost))
                    throw std::runtime_error("frechet_mean: encountered infinite Wasserstein cost during matching");
                matchings[i] = std::move(matching);
            };

            if (params.n_threads <= 1) {
                for (size_t i = 0; i < split_diagrams.size(); ++i)
                    compute_matching(i);
            } else {
                tf::Executor executor(params.n_threads);
                tf::Taskflow flow;
                flow.for_each_index(size_t(0), split_diagrams.size(), size_t(1), compute_matching);
                executor.run(flow).get();
            }

            SplitDiagram<Real> updated_barycenter;
            updated_barycenter.finite.reserve(barycenter_parts.finite.size());

            for (size_t barycenter_idx = 0; barycenter_idx < barycenter_parts.finite.size(); ++barycenter_idx) {
                std::vector<Point> matched_points;
                std::vector<Real> matched_weights;
                matched_points.reserve(split_diagrams.size());
                matched_weights.reserve(split_diagrams.size());
                Real diagonal_weight = static_cast<Real>(0);

                for (size_t diagram_idx = 0; diagram_idx < split_diagrams.size(); ++diagram_idx) {
                    const auto& matching = matchings[diagram_idx].matching_a_to_b_;
                    auto match_iter = matching.find(static_cast<int>(barycenter_idx));
                    if (match_iter == matching.end() || match_iter->second < 0) {
                        diagonal_weight += normalized_weights[diagram_idx];
                    } else {
                        matched_points.push_back(split_diagrams[diagram_idx].finite.at(static_cast<size_t>(match_iter->second)));
                        matched_weights.push_back(normalized_weights[diagram_idx]);
                    }
                }

                auto new_point = update_finite_barycenter_point_q2<Real>(
                        matched_points,
                        matched_weights,
                        diagonal_weight,
                        params.tol);

                if (!new_point.is_diagonal())
                    updated_barycenter.finite.push_back(new_point);
            }

            updated_barycenter.neg_inf_birth = infinite_barycenter.neg_inf_birth;
            updated_barycenter.pos_inf_birth = infinite_barycenter.pos_inf_birth;
            updated_barycenter.neg_inf_death = infinite_barycenter.neg_inf_death;
            updated_barycenter.pos_inf_death = infinite_barycenter.pos_inf_death;

            bool converged = updated_barycenter.finite.size() == barycenter_parts.finite.size();
            if (converged) {
                Real max_change = static_cast<Real>(0);
                for (size_t i = 0; i < barycenter_parts.finite.size(); ++i)
                    max_change = std::max(max_change, convergence_distance<Real>(barycenter_parts.finite[i], updated_barycenter.finite[i]));
                for (int family = 1; family <= 4; ++family) {
                    const auto& current_family = infinite_family_points<Real>(barycenter_parts, family);
                    const auto& updated_family = infinite_family_points<Real>(updated_barycenter, family);
                    for (size_t i = 0; i < current_family.size(); ++i)
                        max_change = std::max(max_change, convergence_distance<Real>(current_family[i], updated_family[i]));
                }
                converged = max_change <= params.tol;
            }

            barycenter_parts = std::move(updated_barycenter);

            if (barycenter_parts.finite.empty()
                    && barycenter_parts.neg_inf_birth.empty()
                    && barycenter_parts.pos_inf_birth.empty()
                    && barycenter_parts.neg_inf_death.empty()
                    && barycenter_parts.pos_inf_death.empty())
                break;

            if (converged)
                break;
        }

        return combine_split_diagram<Real>(std::move(barycenter_parts));
    }

} // namespace oineus
