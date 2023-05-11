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
    std::pair<Simplex<Int, Real>, VREdge> vr_simplex_with_edge(const std::vector<Point<Real, D>>& points, const std::vector<size_t>& vertices_)
    {
        using Simplex = Simplex<Int, Real>;
        using IdxVector = typename Simplex::IdxVector;

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

        return {Simplex(vertices, crit_value), crit_edge};
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
    std::pair<Filtration<Int, Real>, std::vector<VREdge>> get_vr_filtration_bk(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_radius = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using Filtration = Filtration<Int, Real>;
        using Simplex = Simplex<Int, Real>;
        using VertexContainer = std::vector<size_t>;

        auto neighbor = [&](size_t u, size_t v) { return sq_dist(points[u], points[v]) <= max_radius * max_radius; };

        std::vector<Simplex> simplices;
        std::vector<VREdge> edges;
        bool negate {false};

        // vertices are added manually to preserve order (id == index)
        for(size_t v = 0; v < points.size(); ++v) {
            auto [simplex, edge] = vr_simplex_with_edge<Int, Real>(points, {v});
            simplices.emplace_back(simplex);
            edges.emplace_back(edge);
        }

        auto functor = [&](const VertexContainer& vs)
                { if (vs.size() > 1) {
                    auto [simplex, edge] = vr_simplex_with_edge<Int, Real, D>(points, vs);
                    simplices.emplace_back(simplex);
                    edges.emplace_back(edge);
                }
                };

        VertexContainer current;
        VertexContainer candidates(points.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        auto excluded_end {candidates.cbegin()};
        bool check_initial {false};

        bron_kerbosch(current, candidates, excluded_end, max_dim, neighbor, functor, check_initial);

        return std::make_pair(Filtration(std::move(simplices), negate, n_threads), std::move(edges));
    }

    // stupid brute-force
    template<class Int, class Real, std::size_t D>
    decltype(auto) get_vr_filtration_naive(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_radius = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using VRFiltration = Filtration<Int, Real>;
        using VRSimplex = Simplex<Int, Real>;

        std::vector<VRSimplex> simplices;
        std::vector<VREdge> edges;

        for(size_t v_idx = 0; v_idx < points.size(); ++v_idx) {
            std::vector<size_t> vertices {v_idx};
            auto [simplex, edge] = vr_simplex_with_edge<Int, Real, D>(points, vertices);
            simplices.emplace_back(simplex);
            edges.emplace_back(edge);
        }

        if (max_dim >= 1)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx) {
                    auto [s, e] = vr_simplex_with_edge<Int, Real, D>(points, {u_idx, v_idx});
                    if (s.value() <= max_radius) {
                        simplices.emplace_back(s);
                        edges.emplace_back(e);
                    }
                }

        if (max_dim >= 2)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx)
                    for(size_t w_idx = v_idx + 1; w_idx < points.size(); ++w_idx) {
                        auto [s, e] = vr_simplex_with_edge<Int, Real, D>(points, {u_idx, v_idx, w_idx});
                        if (s.value() <= max_radius) {
                            simplices.emplace_back(s);
                            edges.emplace_back(e);
                        }
                    }

        if (max_dim >= 3)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx)
                    for(size_t w_idx = v_idx + 1; w_idx < points.size(); ++w_idx)
                        for(size_t t_idx = w_idx + 1; t_idx < points.size(); ++t_idx) {
                            auto [s, e] = vr_simplex_with_edge<Int, Real, D>(points, {u_idx, v_idx, w_idx, t_idx});
                            if (s.value() <= max_radius) {
                                simplices.emplace_back(s);
                                edges.emplace_back(e);
                            }
                        }

        if (max_dim >= 4)
            throw std::runtime_error("not implemented");

        return std::make_pair<VRFiltration, std::vector<VREdge>>(VRFiltration(std::move(simplices), false, n_threads), std::move(edges));
    }

};

#endif //OINEUS_VIETORIS_RIPS_H
