#ifndef OINEUS_VIETORIS_RIPS_H
#define OINEUS_VIETORIS_RIPS_H

#include <vector>
#include <array>
#include <limits>
#include <cassert>

#include "filtration.h"

namespace oineus {

    template<class Real>
    struct DistMatrix {
        Real* distances;
        size_t n_points;
        Real get_distance(size_t i, size_t j) const { return distances[n_points*i + j]; }
    };

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

    template<class Int, class Real>
    std::pair<CellWithValue<Simplex<Int>, Real>, VREdge<Int>> vr_simplex_with_edge(const DistMatrix<Real>& dist_matrix, const typename Simplex<Int>::IdxVector& vertices)
    {
        using SimplexWithValue = CellWithValue<Simplex<Int>, Real>;

        assert(not vertices.empty());

        Real crit_value = 0;
        VREdge<Int> crit_edge {vertices[0], vertices[0]};

        for(size_t u_idx = 0; u_idx < vertices.size(); ++u_idx) {
            for(size_t v_idx = u_idx + 1; v_idx < vertices.size(); ++v_idx) {
                auto u = vertices[u_idx];
                auto v = vertices[v_idx];
                Real uv_dist = dist_matrix.get_distance(u, v);
                if (uv_dist > crit_value) {
                    crit_value = uv_dist;
                    crit_edge = {u, v};
                }
            }
        }

        // false: do not compute uid immediately, set it in filtration ctor in parallel
        return {SimplexWithValue(Simplex(vertices, false), crit_value), crit_edge};
    }

    template<class Int, class Real>
    CellWithValue<Simplex<Int>, Real> vr_simplex(const DistMatrix<Real>& dist_matrix, const typename Simplex<Int>::IdxVector& vertices)
    {
        using Simp = CellWithValue<Simplex<Int>, Real>;

        assert(not vertices.empty());

        Real crit_value = 0;

        for(size_t u_idx = 0; u_idx < vertices.size(); ++u_idx) {
            for(size_t v_idx = u_idx + 1; v_idx < vertices.size(); ++v_idx) {
                auto u = vertices[u_idx];
                auto v = vertices[v_idx];
                Real uv_dist = dist_matrix.get_distance(u, v);
                if (uv_dist > crit_value) {
                    crit_value = uv_dist;
                }
            }
        }

        // false in Simplex<Int> ctor: do not set uid immediately
        return Simp({vertices, false}, crit_value);
    }

    template<class Int, class Real, std::size_t D>
    std::pair<CellWithValue<Simplex<Int>, Real>, VREdge<Int>> vr_simplex_with_edge(const std::vector<Point<Real, D>>& points, const typename Simplex<Int>::IdxVector& vertices_)
    {
        using Simplex = CellWithValue<Simplex<Int>, Real>;

        assert(not vertices_.empty());

        Real crit_value = 0;
        VREdge<Int> crit_edge {vertices_[0], vertices_[0]};

        for(size_t u_idx = 0; u_idx < vertices_.size(); ++u_idx) {
            for(size_t v_idx = u_idx + 1; v_idx < vertices_.size(); ++v_idx) {
                auto u = vertices_[u_idx];
                auto v = vertices_[v_idx];
                if (sq_dist(points[u], points[v]) > crit_value) {
                    crit_value = sq_dist(points[u], points[v]);
                    crit_edge = {u, v};
                }
            }
        }

        crit_value = sqrt(crit_value);

        return {Simplex({vertices_}, crit_value), crit_edge};
    }

    template<class Int, class Real, std::size_t D>
    CellWithValue<Simplex<Int>, Real> vr_simplex(const std::vector<Point<Real, D>>& points, const typename Simplex<Int>::IdxVector& vertices)
    {
        using Simplex = CellWithValue<Simplex<Int>, Real>;

        assert(not vertices.empty());

        Real crit_value = 0;

        for(size_t u_idx = 0; u_idx < vertices.size(); ++u_idx) {
            for(size_t v_idx = u_idx + 1; v_idx < vertices.size(); ++v_idx) {
                auto u = vertices[u_idx];
                auto v = vertices[v_idx];
                if (sq_dist(points[u], points[v]) > crit_value) {
                    crit_value = sq_dist(points[u], points[v]);
                }
            }
        }

        crit_value = sqrt(crit_value);

        return Simplex({vertices}, crit_value);
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
        CALI_CXX_MARK_FUNCTION;
        if (check_initial and not current.empty())
            functor(current);

        if (current.size() == max_dim + 1)
            return;

        [[maybe_unused]] size_t cur_idx = 0;
        for(auto cur = excluded_end; cur != candidates.end(); ++cur) {
            current.push_back(*cur);

            VertexContainer new_candidates;

            [[maybe_unused]] size_t i = 0;

            for(auto ccur = candidates.begin(); ccur != cur; ++ccur) {
                if (neighbor(*ccur, *cur))
                    new_candidates.push_back(*ccur);

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                if (i % 500 == 0) {
                    OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                }
                i++;
#endif
            }

            size_t ex = new_candidates.size();

            i = 0;
            for(auto ccur = std::next(cur); ccur != candidates.end(); ++ccur) {
                if (neighbor(*ccur, *cur))
                    new_candidates.push_back(*ccur);

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                if (i % 500 == 0) {
                    OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                }
                i++;
#endif

            }

            excluded_end = new_candidates.begin() + ex;

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
            if (cur_idx % 500 == 0) {
                OINEUS_CHECK_FOR_PYTHON_INTERRUPT
            }
            cur_idx++;
#endif

            bron_kerbosch(current, new_candidates, excluded_end, max_dim, neighbor, functor, true);

            current.pop_back();
        }
    }


    // Bron-Kerbosch, from Dionysus
    template<class Int, class Real, std::size_t D>
    std::pair<Filtration<Simplex<Int>, Real>, std::vector<VREdge<Int>>> get_vr_filtration_and_critical_edges(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_diameter = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        CALI_CXX_MARK_FUNCTION;
        using VRFiltration = Filtration<Simplex<Int>, Real>;
        using Simplex = typename VRFiltration::Cell;
        using VertexContainer = typename Simplex::Cell::IdxVector;
        using Edges = std::vector<VREdge<Int>>;

        auto neighbor = [&](size_t u, size_t v) { return sq_dist(points[u], points[v]) <= max_diameter * max_diameter; };

        std::vector<Simplex> simplices;
        Edges edges;
        bool negate {false};

        // vertices are added manually to preserve order (id == index)
        for(Int v = 0; v < static_cast<Int>(points.size()); ++v) {
            auto [simplex, edge] = vr_simplex_with_edge<Int, Real, D>(points, {v});
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
        // Filtration constructor will sort simplices and assign sorted ids
        auto fil = Filtration(std::move(simplices), negate, n_threads);

        // use sorted info from fil to rearrange edges
        Edges sorted_edges;

        sorted_edges.reserve(edges.size());

        for(size_t sorted_edge_idx = 0; sorted_edge_idx < edges.size(); ++sorted_edge_idx) {
            sorted_edges.push_back(edges[fil.get_id_by_sorted_id(sorted_edge_idx)]);
        }

        return std::make_pair(fil, sorted_edges);
    }

    template<class Int, class Real>
    std::pair<Filtration<Simplex<Int>, Real>, std::vector<VREdge<Int>>> get_vr_filtration_and_critical_edges(const DistMatrix<Real>& dist_matrix, dim_type max_dim, Real max_diameter = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using Filtration = Filtration<Simplex<Int>, Real>;
        using VertexContainer = typename Simplex<Int>::IdxVector;
        using Simplex = CellWithValue<Simplex<Int>, Real>;

        auto neighbor = [&](size_t u, size_t v) { return dist_matrix.get_distance(u, v) <= max_diameter * max_diameter; };

        std::vector<Simplex> simplices;
        std::vector<VREdge<Int>> edges;
        bool negate {false};

        Int n_points = dist_matrix.n_points;

        // vertices are added manually to preserve order (id == index)
        for(Int v = 0; v < n_points; ++v) {
            auto [simplex, edge] = vr_simplex_with_edge<Int, Real>(dist_matrix, {v});
            simplices.emplace_back(simplex);
            edges.emplace_back(edge);
        }


        auto functor = [&](const VertexContainer& vs)
                { if (vs.size() > 1) {
                    auto [simplex, edge] = vr_simplex_with_edge<Int, Real>(dist_matrix, vs);
                    simplices.emplace_back(simplex);
                    edges.emplace_back(edge);
                }
                };

        VertexContainer current;
        VertexContainer candidates(n_points);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto excluded_end {candidates.cbegin()};
        bool check_initial {false};

        bron_kerbosch(current, candidates, excluded_end, max_dim, neighbor, functor, check_initial);
        // Filtration constructor will sort simplices and assign sorted ids
        auto fil = Filtration(std::move(simplices), negate, n_threads);

        // use sorted info from fil to rearrange edges
        std::vector<VREdge<Int>> sorted_edges;
        sorted_edges.reserve(edges.size());

        for(size_t sorted_edge_idx = 0; sorted_edge_idx < edges.size(); ++sorted_edge_idx) {
            sorted_edges.push_back(edges[fil.get_id_by_sorted_id(sorted_edge_idx)]);
        }

        return std::make_pair(fil, sorted_edges);
    }

    template<class Int, class Real, std::size_t D>
    auto get_vr_filtration(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_diameter = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        CALI_CXX_MARK_FUNCTION;
        using VertexContainer = typename Simplex<Int>::IdxVector;

        auto neighbor = [&](size_t u, size_t v) { return sq_dist(points[u], points[v]) <= max_diameter * max_diameter; };

        std::vector<CellWithValue<Simplex<Int>, Real>> simplices;
        bool negate {false};

        // vertices are added manually to preserve order (id == index)
        for(Int v = 0; v < static_cast<Int>(points.size()); ++v) {
            simplices.emplace_back(vr_simplex<Int, Real, D>(points, {v}));
        }

        auto functor = [&](const VertexContainer& vs)
                { if (vs.size() > 1) {
                    simplices.emplace_back(vr_simplex<Int, Real, D>(points, vs));
                }
                };

        VertexContainer current;
        VertexContainer candidates(points.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        auto excluded_end {candidates.cbegin()};
        bool check_initial {false};

        bron_kerbosch(current, candidates, excluded_end, max_dim, neighbor, functor, check_initial);
        // Filtration constructor will sort simplices and assign sorted ids
        return Filtration<Simplex<Int>, Real>(std::move(simplices), negate, n_threads);
    }

    template<class Int, class Real>
    auto get_vr_filtration(const DistMatrix<Real>& dist_matrix, dim_type max_dim, Real max_diameter = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using Filtration = Filtration<Simplex<Int>, Real>;
        using VertexContainer = typename Simplex<Int>::IdxVector;
        using Simplex = CellWithValue<Simplex<Int>, Real>;

        auto neighbor = [&](size_t u, size_t v) { return dist_matrix.get_distance(u, v) <= max_diameter * max_diameter; };

        std::vector<Simplex> simplices;
        bool negate {false};

        Int n_points = dist_matrix.n_points;

        // vertices are added manually to preserve order (id == index)
        for(Int v = 0; v < n_points; ++v) {
            auto simplex = vr_simplex<Int, Real>(dist_matrix, {v});
            simplices.emplace_back(simplex);
        }

        auto functor = [&](const VertexContainer& vs)
                { if (vs.size() > 1) {
                    simplices.emplace_back(vr_simplex<Int, Real>(dist_matrix, vs));
                }
                };

        VertexContainer current;
        VertexContainer candidates(n_points);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto excluded_end {candidates.cbegin()};
        bool check_initial {false};

        bron_kerbosch(current, candidates, excluded_end, max_dim, neighbor, functor, check_initial);
        // Filtration constructor will sort simplices and assign sorted ids
        return Filtration(std::move(simplices), negate, n_threads);
    }


    // stupid brute-force
    template<class Int, class Real, std::size_t D>
    std::pair<Filtration<Simplex<Int>, Real>, std::vector<VREdge<Int>>> get_vr_filtration_and_critical_edges_naive(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_diameter = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        using VRFiltration = Filtration<Simplex<Int>, Real>;
        using VRSimplex = typename VRFiltration::Cell;

        std::vector<VRSimplex> simplices;
        std::vector<VREdge<Int>> edges;

        for(size_t v_idx = 0; v_idx < points.size(); ++v_idx) {
            std::vector<Int> vertices {static_cast<Int>(v_idx)};
            auto [simplex, edge] = vr_simplex_with_edge<Int, Real, D>(points, vertices);
            simplices.emplace_back(simplex);
            edges.emplace_back(edge);
        }

        if (max_dim >= 1)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx) {
                    auto [s, e] = vr_simplex_with_edge<Int, Real, D>(points, {(Int)u_idx, (Int)v_idx});
                    if (s.get_value() <= max_diameter) {
                        simplices.emplace_back(s);
                        edges.emplace_back(e);
                    }
                }

        if (max_dim >= 2)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx)
                    for(size_t w_idx = v_idx + 1; w_idx < points.size(); ++w_idx) {
                        auto [s, e] = vr_simplex_with_edge<Int, Real, D>(points, {(Int)u_idx, (Int)v_idx, (Int)w_idx});
                        if (s.get_value() <= max_diameter) {
                            simplices.emplace_back(s);
                            edges.emplace_back(e);
                        }
                    }

        if (max_dim >= 3)
            for(size_t u_idx = 0; u_idx < points.size(); ++u_idx)
                for(size_t v_idx = u_idx + 1; v_idx < points.size(); ++v_idx)
                    for(size_t w_idx = v_idx + 1; w_idx < points.size(); ++w_idx)
                        for(size_t t_idx = w_idx + 1; t_idx < points.size(); ++t_idx) {
                            auto [s, e] = vr_simplex_with_edge<Int, Real, D>(points, {(Int)u_idx, (Int)v_idx, (Int)w_idx, (Int)t_idx});
                            if (s.get_value() <= max_diameter) {
                                simplices.emplace_back(s);
                                edges.emplace_back(e);
                            }
                        }

        if (max_dim >= 4)
            throw std::runtime_error("not implemented");

        return std::make_pair<VRFiltration, std::vector<VREdge<Int>>>(VRFiltration(std::move(simplices), false, n_threads), std::move(edges));
    }

    template<class Int, class Real, std::size_t D>
    auto get_vr_filtration_naive(const std::vector<Point<Real, D>>& points, dim_type max_dim = D, Real max_diameter = std::numeric_limits<Real>::max(), int n_threads = 1)
    {
        return get_vr_filtration_and_critical_edges_naive<Int, Real, D>(points, max_dim, max_diameter, n_threads).first;
    }

};

#endif //OINEUS_VIETORIS_RIPS_H
