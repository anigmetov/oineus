#ifndef OINEUS_VIETORIS_RIPS_H
#define OINEUS_VIETORIS_RIPS_H

#include <vector>
#include <array>
#include <limits>
#include <cassert>

#include "filtration.h"

namespace oineus {

    template<class Real, std::size_t D>
    using Point = std::array<Real, D>;

    template<class Real, std::size_t D>
    Real sq_dist(const Point<Real, D>& a, const Point<Real, D>& b)
    {
        Real res = 0;
        for(size_t i = 0; i < D; ++i)
            res += (a[i] - b[i]) * (a[i] - b[i]);
        return res;
    }

    template<class Real, std::size_t D>
    Real dist(const Point<Real, D>& a, const Point<Real, D>& b)
    {
        return sqrt(sq_dist(a, b));
    }

    template<class Int, class Real, std::size_t D>
    Simplex<Int, Real, VREdge> vr_simplex(const std::vector<Point<Real, D>>& points, const std::vector<size_t>& vertices_)
    {
        using VRSimplex = Simplex<Int, Real, VREdge>;
        using IdxVector = typename VRSimplex::IdxVector;

        assert(not vertices_.empty());

        Real crit_value = 0;
        VREdge crit_edge {vertices_[0], vertices_[0]};

        for(size_t u_idx = 0; u_idx < vertices_.size(); ++u_idx) {
            for(size_t v_idx = u_idx + 1; v_idx < vertices_.size(); ++v_idx) {
                size_t u = vertices_[u_idx];
                size_t v = vertices_[v_idx];
                if (sq_dist(points[u], points[v]) > crit_value) {
                    crit_value = sq_dist(points[u], points[v]);
                    crit_edge = {u, v};
                }
            }
        }

        crit_value = sqrt(crit_value);

        // convert size_t to Int, if necessary
        IdxVector vertices {vertices_.begin(), vertices_.end()};

        return VRSimplex(vertices, crit_value, crit_edge);
    }

    template<class Functor, class NeighborTest, class VertexContainer>
    void bron_kerbosch(VertexContainer& current,
            const VertexContainer& candidates,
            typename VertexContainer::const_iterator excluded_end,
            dim_type max_dim,
            const NeighborTest& neighbor,
            const Functor& functor,
            bool check_initial)
    {
        if (check_initial and not current.empty())
            functor(current);

        if (current.size() == max_dim + 1)
            return;

        for(auto cur = excluded_end; cur != candidates.end(); ++cur) {
            current.push_back(*cur);

            VertexContainer new_candidates;

            for(auto ccur = candidates.begin(); ccur != cur; ++ccur)
                if (neighbor(*ccur, *cur))
                    new_candidates.push_back(*ccur);

            size_t ex = new_candidates.size();

            for(auto ccur = std::next(cur); ccur != candidates.end(); ++ccur)
                if (neighbor(*ccur, *cur))
                    new_candidates.push_back(*ccur);

            excluded_end = new_candidates.begin() + ex;

            bron_kerbosch(current, new_candidates, excluded_end, max_dim, neighbor, functor, true);

            current.pop_back();
        }
    }

// Bron-Kerbosch, from Dionysus
    template<class Int, class Real, std::size_t D>
    decltype(auto) get_vr_filtration_bk(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_radius = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using VRFiltration = Filtration<Int, Real, VREdge>;
        using VRSimplex = Simplex<Int, Real, VREdge>;
        using VertexContainer = std::vector<size_t>;

        auto neighbor = [&](size_t u, size_t v) { return sq_dist(points[u], points[v]) <= max_radius * max_radius; };

        std::vector<VRSimplex> simplices;
        bool negate {false};

        // vertices are added manually to preserve order (id == index)
        for(size_t v = 0; v < points.size(); ++v)
            simplices.emplace_back(vr_simplex<Int, Real>(points, {v}));

        auto functor = [&](const VertexContainer& vs) { if (vs.size() > 1) simplices.push_back(vr_simplex<Int, Real, D>(points, vs)); };

        VertexContainer current;
        VertexContainer candidates(points.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        auto excluded_end {candidates.cbegin()};
        bool check_initial {false};

        bron_kerbosch(current, candidates, excluded_end, max_dim, neighbor, functor, check_initial);

        return VRFiltration(simplices, negate, n_threads);
    }

// stupid brute-force
    template<class Int, class Real, std::size_t D>
    decltype(auto) get_vr_filtration_naive(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_radius = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using VRFiltration = Filtration<Int, Real, VREdge>;
        using VRSimplex = Simplex<Int, Real, VREdge>;

        std::vector<VRSimplex> simplices;

        for(size_t v_idx = 0; v_idx < points.size(); ++v_idx) {
            std::vector<size_t> vertices {v_idx};
            simplices.emplace_back(vr_simplex<Int, Real, D>(points, vertices));
        }

        if (max_dim >= 1)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx) {
                    auto s = vr_simplex<Int, Real, D>(points, {u_idx, v_idx});
                    if (s.value() <= max_radius)
                        simplices.emplace_back(s);
                }

        if (max_dim >= 2)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx)
                    for(size_t w_idx = v_idx + 1; w_idx < points.size(); ++w_idx) {
                        auto s = vr_simplex<Int, Real, D>(points, {u_idx, v_idx, w_idx});
                        if (s.value() <= max_radius)
                            simplices.emplace_back(s);
                    }

        if (max_dim >= 3)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx)
                    for(size_t w_idx = v_idx + 1; w_idx < points.size(); ++w_idx)
                        for(size_t t_idx = w_idx + 1; t_idx < points.size(); ++t_idx) {
                            auto s = vr_simplex<Int, Real, D>(points, {u_idx, v_idx, w_idx, t_idx});
                            if (s.value() <= max_radius)
                                simplices.emplace_back(s);
                        }

        if (max_dim >= 4)
            throw std::runtime_error("not implemented");

        return VRFiltration(simplices, false, n_threads);
    }

};

#endif //OINEUS_VIETORIS_RIPS_H
