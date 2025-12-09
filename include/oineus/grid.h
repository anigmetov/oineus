#ifndef OINEUS_GRID_H
#define OINEUS_GRID_H

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <thread>
#include <utility>
#include <tuple>

#include "grid_domain.h"
#include "filtration.h"
#include "simplex.h"
#include "cube.h"
#include "timer.h"

namespace oineus {

template<typename Int_, typename Real_, size_t D>
class Grid {
public:
    using Domain = GridDomain<Int_, D>;
    using Int = Int_;
    using Real = Real_;
    using GridPoint = typename Domain::GridPoint;
    using GridPointVec = typename Domain::GridPointVec;
    using GridPointVecVec = typename Domain::GridPointVecVec;

    enum class DataLocation { VERTEX, CELL};

    struct ValueVertex {
        Real value;
        Int vertex;
    };

    using GridSimplex = CellWithValue<Simplex<Int>, Real>;
    using GridCube = Cube<Int, D>;
    using GridCubeVal = CellWithValue<Cube<Int, D>, Real>;
    using SimplexVec = std::vector<GridSimplex>;
    using GridCubeVec = std::vector<GridCubeVal>;
    using CriticalIndices = std::vector<Int>;
    using IdxVector = typename Simplex<Int>::IdxVector;
    using GridFiltration = Filtration<Simplex<Int>, Real>;
    using GridCubeFiltration = Filtration<Cube<Int, D>, Real>;

    static constexpr std::size_t dim {D};

    Grid() = default;
    Grid(const Grid&) = default;
    Grid(Grid&&) noexcept = default;
    Grid& operator=(const Grid&) = default;
    Grid& operator=(Grid&&) noexcept = default;

    Grid(const GridPoint& _dims, bool _wrap, const Real* const _data, DataLocation _data_location)
            : data_location_(_data_location), data_domain_(_dims, _wrap), data_(_data)
    {
        if (data_location_ == DataLocation::VERTEX) {
            computational_domain_ = data_domain_;
        } else {
            GridPoint comp_dims = _dims;
            for(size_t d = 0; d < dim; ++d) { comp_dims[d] += 1; }
            computational_domain_ = Domain(comp_dims, _wrap);
        }
    }

    std::string data_location_as_string() const { if (data_location_ == DataLocation::CELL) return "cells"; else return "vertices"; }

    // wrappers for Domain
    Int size() const { return computational_domain_.size(); }
    Int point_to_id(const GridPoint& v) const { return computational_domain_.point_to_id(v); }
    GridPoint id_to_point(Int i) const { return computational_domain_.id_to_point(i); }
    GridPoint wrap_point(const GridPoint& v) const { return computational_domain_.wrap_point(v); }
    bool contains(const GridPoint& v) const { return computational_domain_.contains(v); }

    GridPointVecVec get_fr_displacements(size_t d) const {
        if (data_location_ == DataLocation::VERTEX) {
            return computational_domain_.get_fr_displacements(d);
        } else {
            throw std::runtime_error("Freudenthal triangulation requires data on vertices");
        }
    }

    Int get_n_cubes_in_dimension(dim_type d) const { return computational_domain_.get_n_cubes_in_dimension(d); }

    template<class Cont>
    static GridPoint add_points(const GridPoint& x, const Cont& y) { return Domain::add_points(x, y); }

    Real value_at_vertex(Int vertex) const
    {
        // assumes C order
        assert(vertex >= 0 and vertex < size());
        return *(data_ + vertex);
    }

    Real value_at_vertex(const GridPoint& vertex) const
    {
        // assumes C order
        auto vertex_idx = data_domain_.point_to_id(vertex);
        return *(data_ + vertex_idx);
    }

    std::pair<GridFiltration, CriticalIndices> freudenthal_filtration_and_critical_vertices(size_t top_d, bool negate, int n_threads = 1) const
    {
        if (top_d > dim)
            throw std::runtime_error("bad dimension, top_d = " + std::to_string(top_d) + ", dim = " + std::to_string(dim));

        SimplexVec simplices;
        CriticalIndices vertices;

        // calculate total number of cells to allocate memory once
        size_t total_size = 0;
        for(dim_type d = 0 ; d <= top_d ; ++d) {
            total_size += get_fr_displacements(d).size() * size();
        }

        simplices.reserve(total_size);
        vertices.reserve(total_size);

        for(dim_type d = 0 ; d <= top_d ; ++d) {
            add_freudenthal_simplices(d, negate, simplices, vertices, true);
        }

        auto fil = GridFiltration(simplices, negate, n_threads);

        CriticalIndices sorted_vertices;

        for(const auto& cell: fil.cells()) {
            sorted_vertices.push_back(vertices[cell.get_id()]);
        }

        return {fil, sorted_vertices};
    }

    GridFiltration freudenthal_filtration(size_t top_d, bool negate, int n_threads = 1) const
    {
        if (top_d > dim)
            throw std::runtime_error("bad dimension, top_d = " + std::to_string(top_d) + ", dim = " + std::to_string(dim));

        SimplexVec simplices;
        CriticalIndices dummy_vertices;

        // calculate total number of cells to allocate memory once
        size_t total_size = 0;
        for(dim_type d = 0 ; d <= top_d ; ++d) {
            total_size += get_fr_displacements(d).size() * size();
        }

        simplices.reserve(total_size);

        for(dim_type d = 0 ; d <= top_d ; ++d) {
            add_freudenthal_simplices(d, negate, simplices, dummy_vertices, false);
        }

        return GridFiltration(simplices, negate, n_threads);
    }

    const Domain& domain() const { return computational_domain_; }
    const Domain& data_domain() const { return data_domain_; }

    ValueVertex simplex_value_and_vertex(const IdxVector& vertices, bool negate) const
    {
        auto cmp = [negate](Real a, Real b) { return negate ? a > b : a < b; };

        ValueVertex result {negate ? std::numeric_limits<Real>::max() : std::numeric_limits<Real>::lowest(), Int(-1)};

        for(Int v: vertices) {
            auto v_value = value_at_vertex(v);
            if (cmp(result.value, v_value))
                result = {v_value, v};
        }

        return result;
    }

    GridCubeFiltration cube_filtration(size_t top_d, bool negate, int n_threads = 1) const
    {
        bool verbose = false;
        Timer timer;
        if (top_d > dim)
            throw std::runtime_error("bad dimension, top_d = " + std::to_string(top_d) + ", dim = " + std::to_string(dim));

        GridCubeVec cubes;

        size_t total_size = 0;

        for(dim_type d = 0 ; d <= top_d ; ++d) {
            total_size += get_n_cubes_in_dimension(d) * size();
        }

        if (n_threads == 1) {
            // calculate total number of cells to allocate memory once
            cubes.reserve(total_size);

            for(Int v = 0 ; v < size() ; ++v) {
                Int v_part = v << OINEUS_MAX_CUBE_DIM;
                for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                    Int cube_id = v_part | face_part;
                    GridCube cube = GridCube(cube_id, computational_domain_);

                    auto [is_valid, cube_value] = get_cube_validity_and_value(cube, negate);

                    if (not is_valid)
                        continue;

                    cubes.emplace_back(cube, cube_value);

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                    if (v % 100 == 0) {
                        OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
                    }
#endif
                }
            }
        } else {
            timer.reset();
            // Multi-threaded version
            Int n_vertices = size();
            std::vector<GridCubeVec> thread_cubes(n_threads);

            // Pre-allocate approximate memory for each thread
            size_t approx_per_thread = total_size / n_threads + 1;
            for (auto& tc : thread_cubes) {
                tc.reserve(approx_per_thread);
            }

            // Launch threads
            std::vector<std::thread> threads;
            threads.reserve(n_threads);

            for (int t = 0; t < n_threads; ++t) {
                threads.emplace_back([this, t, n_threads, n_vertices, negate, &thread_cubes]() {
                    auto& local_cubes = thread_cubes[t];

                    // Calculate contiguous chunk for this thread
                    Int chunk_size = n_vertices / n_threads;
                    Int v_start = t * chunk_size;
                    Int v_end = (t == n_threads - 1) ? n_vertices : (t + 1) * chunk_size;

                    for (Int v = v_start; v < v_end; ++v) {
                        Int v_part = v << OINEUS_MAX_CUBE_DIM;
                        for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                            Int cube_id = v_part | face_part;
                            GridCube cube = GridCube(cube_id, computational_domain_);
                            auto [is_valid, cube_value] = get_cube_validity_and_value(cube, negate);
                            if (is_valid) {
                                local_cubes.emplace_back(std::move(cube), cube_value);
                            }
                        }
                    }
                });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }

            auto cube_elapsed = timer.elapsed_reset();

            // Combine all thread-local vectors
            size_t final_size = 0;
            for (const auto& tc : thread_cubes) {
                final_size += tc.size();
            }
            cubes.reserve(final_size);

            for (auto& tc : thread_cubes) {
                cubes.insert(cubes.end(),
                            std::make_move_iterator(tc.begin()),
                            std::make_move_iterator(tc.end()));
            }

            auto move_elapsed = timer.elapsed_reset();
            if (verbose) {
                std::cerr << "cube_elapsed : " << cube_elapsed << "\n";
                std::cerr << "move_elapsed : " << move_elapsed << "\n";
            }
        }

        timer.reset();
        auto fil = GridCubeFiltration(std::move(cubes), negate, n_threads, false);
        auto fil_elapsed = timer.elapsed_reset();
        if (verbose)
            std::cerr << "fil_elapsed : " << fil_elapsed << "\n";
        return fil;
    }

    std::pair<GridCubeFiltration, CriticalIndices> cube_filtration_and_critical_indices(size_t top_d, bool negate, int n_threads = 1) const
    {
        bool verbose = false;
        Timer timer;
        if (top_d > dim)
            throw std::runtime_error("bad dimension, top_d = " + std::to_string(top_d) + ", dim = " + std::to_string(dim));

        GridCubeVec cubes;
        CriticalIndices critical_indices;

        size_t total_size = 0;

        for(dim_type d = 0 ; d <= top_d ; ++d) {
            total_size += get_n_cubes_in_dimension(d) * size();
        }

        if (n_threads == 1) {
            // calculate total number of cells to allocate memory once
            cubes.reserve(total_size);
            critical_indices.reserve(total_size);

            for(Int v = 0 ; v < size() ; ++v) {
                Int v_part = v << OINEUS_MAX_CUBE_DIM;
                for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                    Int cube_id = v_part | face_part;
                    GridCube cube = GridCube(cube_id, computational_domain_);

                    auto [is_valid, cube_value, critical_index] = get_cube_validity_value_and_crit_index(cube, negate);

                    if (not is_valid)
                        continue;

                    cubes.emplace_back(cube, cube_value);
                    critical_indices.emplace_back(critical_index);

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                    if (v % 100 == 0) {
                        OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
                    }
#endif
                }
            }
        } else {
            timer.reset();
            // Multi-threaded version
            Int n_vertices = size();
            std::vector<GridCubeVec> thread_cubes(n_threads);
            std::vector<CriticalIndices> thread_crit_indices(n_threads);

            // Pre-allocate approximate memory for each thread
            size_t approx_per_thread = total_size / n_threads + 1;
            for (auto& tc : thread_cubes) {
                tc.reserve(approx_per_thread);
            }
            for(auto& tci : thread_crit_indices) {
                tci.reserve(approx_per_thread);
            }

            // Launch threads
            std::vector<std::thread> threads;
            threads.reserve(n_threads);

            for (int t = 0; t < n_threads; ++t) {
                threads.emplace_back([this, t, n_threads, n_vertices, negate, &thread_cubes, &thread_crit_indices]() {
                    auto& local_cubes = thread_cubes[t];
                    auto& local_crit_indices = thread_crit_indices[t];

                    // Calculate contiguous chunk for this thread
                    Int chunk_size = n_vertices / n_threads;
                    Int v_start = t * chunk_size;
                    Int v_end = (t == n_threads - 1) ? n_vertices : (t + 1) * chunk_size;

                    for (Int v = v_start; v < v_end; ++v) {
                        Int v_part = v << OINEUS_MAX_CUBE_DIM;
                        for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                            Int cube_id = v_part | face_part;
                            GridCube cube = GridCube(cube_id, computational_domain_);
                            auto [is_valid, cube_value, crit_index] = get_cube_validity_value_and_crit_index(cube, negate);
                            if (is_valid) {
                                local_cubes.emplace_back(std::move(cube), cube_value);
                                local_crit_indices.emplace_back(crit_index);
                            }
                        }
                    }
                });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }

            auto cube_elapsed = timer.elapsed_reset();

            // Combine all thread-local vectors
            size_t final_size = 0;
            for (const auto& tc : thread_cubes) {
                final_size += tc.size();
            }
            cubes.reserve(final_size);

            for (auto& tc : thread_cubes) {
                cubes.insert(cubes.end(),
                            std::make_move_iterator(tc.begin()),
                            std::make_move_iterator(tc.end()));
            }

            for (auto& tci : thread_crit_indices) {
                critical_indices.insert(critical_indices.end(),
                            std::make_move_iterator(tci.begin()),
                            std::make_move_iterator(tci.end()));
            }
            auto move_elapsed = timer.elapsed_reset();
            if (verbose) {
                std::cerr << "cube_elapsed : " << cube_elapsed << "\n";
                std::cerr << "move_elapsed : " << move_elapsed << "\n";
            }
        }

        timer.reset();
        auto fil = GridCubeFiltration(std::move(cubes), negate, n_threads, false);
        auto fil_elapsed = timer.elapsed_reset();
        std::cerr << "fil_elapsed : " << fil_elapsed << "\n";
        return {fil, critical_indices};
    }

    template<typename I, typename R, size_t DD>
    friend std::ostream& operator<<(std::ostream& out, const Grid<I, R, DD>& g);

private:
    DataLocation data_location_;
    Domain data_domain_;           // always corresponds to shape of data
    Domain computational_domain_;  // equals data_domain, if location is VERTEX, else is expanded by 1
    const Real* const data_ {nullptr};

    void add_freudenthal_simplices_from_vertex(const GridPoint& v,
            size_t d,
            bool negate,
            const GridPointVecVec& disps,
            SimplexVec& simplices,
            CriticalIndices& critical_vertices,
            bool return_critical_vertices) const
    {
        IdxVector v_ids(d + 1, 0);

        for(auto& deltas: disps) {
            assert(deltas.size() == d + 1);

            bool is_valid_simplex = true;

            for(size_t i = 0 ; i < d + 1 ; ++i) {
                GridPoint u = add_points(v, deltas[i]);
                if (computational_domain_.wrap())
                    v_ids[i] = point_to_id(wrap_point(u));
                else if (not contains(u)) {
                    is_valid_simplex = false;
                    break;
                } else
                    v_ids[i] = point_to_id(u);
            }

            if (is_valid_simplex) {
                ValueVertex vv = simplex_value_and_vertex(v_ids, negate);
                simplices.emplace_back(v_ids, vv.value);
                if (return_critical_vertices) {
                    critical_vertices.emplace_back(vv.vertex);
                }
            }
        }
    }

    void add_freudenthal_simplices(dim_type d,
                                   bool negate,
                                   SimplexVec& simplices,
                                   CriticalIndices& critical_vertices,
                                   bool return_critical_vertices) const
    {
        auto disps = get_fr_displacements(d);

        for(Int i = 0 ; i < size() ; ++i) {
            GridPoint v = id_to_point(i);
            add_freudenthal_simplices_from_vertex(v, d, negate, disps, simplices, critical_vertices, return_critical_vertices);

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
            if (i % 100 == 0) {
                OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
            }
#endif
        }
    }

    bool is_cell_centric() const { return data_location_ == DataLocation::CELL; }

    Real initial_critical_value(bool negate) const
    {
        // suppose negate is false, we sweep from -inf to +inf (lower-star)
        // if data is on cells, we initialize value with +inf, as we go from top-dim
        // cubes to lower-dimensional ones and apply min()
        // if data is on vertices, we initialize with -inf, as we go from lower-dim
        // cubes to higher-dimensional cubes and apply max()

        bool minus_inf = (is_cell_centric() and negate) or (not is_cell_centric() and not negate);

        if (minus_inf) {
            return -std::numeric_limits<Real>::max();
        } else {
            return std::numeric_limits<Real>::max();
        }
    }

    bool is_cube_valid(const GridCube& cube) const
    {
        if (not computational_domain_.wrap()) {
            for(auto v : cube.get_vertices()) {
                if (not computational_domain_.contains(v)) {
                    return false;
                }
            }
            return true;
        } else {
            throw std::runtime_error("wrap not implemented yet");
        }
    }

    std::tuple<bool, Real, Int> get_cube_validity_value_and_crit_index(const GridCube& cube, bool negate) const
    {
        auto cmp = [negate](Real a, Real b) { return negate ? a > b : a < b; };

        bool is_valid = is_cube_valid(cube);
        Real value = initial_critical_value(negate);
        Int critical_index = k_invalid_index;

        if (is_valid) {
            if (is_cell_centric()) {
                for(auto top_cube_uid: cube.top_cofaces()) {
                    GridCube top_cube(top_cube_uid, computational_domain_);
                    auto anchor_index = data_domain_.point_to_id(top_cube.anchor_vertex());
                    Real top_value = value_at_vertex(anchor_index);

                    if (cmp(top_value, value)) {
                        value = top_value;
                        critical_index = anchor_index;
                    }
                }
            } else {
                for(auto u_local : cube.get_vertices()) {
                    Real u_value = value_at_vertex(u_local);

                    if (cmp(value, u_value)) {
                        value = u_value;
                        critical_index = data_domain_.point_to_id(u_local);
                    }
                }
            }
        }

        return {is_valid, value, critical_index};
    }

    std::pair<bool, Real> get_cube_validity_and_value(const GridCube& cube, bool negate) const
    {
        auto cmp = [negate](Real a, Real b) { return negate ? a > b : a < b; };

        bool is_valid = is_cube_valid(cube);
        Real value = initial_critical_value(negate);

        if (is_valid) {
            if (is_cell_centric()) {
                for(auto top_cube_uid: cube.top_cofaces()) {
                    GridCube top_cube(top_cube_uid, computational_domain_);
                    Real top_value = value_at_vertex(top_cube.anchor_vertex());
                    if (cmp(top_value, value)) {
                        value = top_value;
                    }
                }
            } else {
                for(auto u_local : cube.get_vertices()) {
                    Real u_value = value_at_vertex(u_local);
                    if (cmp(value, u_value))
                        value = u_value;
                }
            }
        }

        return {is_valid, value};
    }

}; // class Grid

template<typename Int, typename Real, size_t D>
std::ostream& operator<<(std::ostream& out, const Grid<Int, Real, D>& g)
{
    out << "Grid(" << g.domain_ << ", data_location=" << g.data_location_as_string() << ", data = " << g.data_ << ")";
    return out;
}

}
#endif //OINEUS_GRID_H