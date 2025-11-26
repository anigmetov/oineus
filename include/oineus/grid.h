#ifndef OINEUS_GRID_H
#define OINEUS_GRID_H

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <thread>

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

    struct ValueVertex {
        Real value;
        Int vertex;
    };

    using GridSimplex = CellWithValue<Simplex<Int>, Real>;
    using GridCube = CellWithValue<Cube<Int, D>, Real>;
    using SimplexVec = std::vector<GridSimplex>;
    using GridCubeVec = std::vector<GridCube>;
    using CriticalVertices = std::vector<Int>;
    using IdxVector = typename Simplex<Int>::IdxVector;
    using GridFiltration = Filtration<Simplex<Int>, Real>;
    using GridCubeFiltration = Filtration<Cube<Int, D>, Real>;

    static constexpr std::size_t dim {D};

    Grid() = default;
    Grid(const Grid&) = default;
    Grid(Grid&&) noexcept = default;
    Grid& operator=(const Grid&) = default;
    Grid& operator=(Grid&&) noexcept = default;

    Grid(const GridPoint& _dims, bool _wrap, Real* _data)
            : domain_(_dims, _wrap), data_(_data) {}

    // wrappers for Domain
    Int size() const { return domain_.size(); }
    Int point_to_id(const GridPoint& v) const { return domain_.point_to_id(v); }
    GridPoint id_to_point(Int i) const { return domain_.id_to_point(i); }
    GridPoint wrap_point(const GridPoint& v) const { return domain_.wrap_point(v); }
    bool contains(const GridPoint& v) const { return domain_.contains(v); }
    GridPointVecVec get_fr_displacements(size_t d) const { return domain_.get_fr_displacements(d); }
    Int get_n_cubes_in_dimension(dim_type d) const { return domain_.get_n_cubes_in_dimension(d); }

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
        auto vertex_idx = domain_.point_to_id(vertex);
        return *(data_ + vertex_idx);
    }

    std::pair<GridFiltration, CriticalVertices> freudenthal_filtration_and_critical_vertices(size_t top_d, bool negate, int n_threads = 1) const
    {
        if (top_d > dim)
            throw std::runtime_error("bad dimension, top_d = " + std::to_string(top_d) + ", dim = " + std::to_string(dim));

        SimplexVec simplices;
        CriticalVertices vertices;

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

        CriticalVertices sorted_vertices;

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
        CriticalVertices dummy_vertices;

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

    const Domain& domain() const { return domain_; }

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

    GridCubeFiltration cube_filtration(size_t top_d, bool negate, bool cell_centric=false, int n_threads = 1) const
    {
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
                Int v_part = v << dim;
                for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                    Int cube_id = v_part | face_part;

                    auto [is_valid, cube_value] = get_cube_validity_and_value(cube_id, negate, cell_centric);

                    if (not is_valid)
                        continue;

                    cubes.emplace_back(Cube<Int, D>(cube_id, domain()), cube_value);

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
                threads.emplace_back([this, t, n_threads, n_vertices, negate, cell_centric, &thread_cubes]() {
                    auto& local_cubes = thread_cubes[t];

                    // Calculate contiguous chunk for this thread
                    Int chunk_size = n_vertices / n_threads;
                    Int v_start = t * chunk_size;
                    Int v_end = (t == n_threads - 1) ? n_vertices : (t + 1) * chunk_size;

                    for (Int v = v_start; v < v_end; ++v) {
                        Int v_part = v << dim;
                        for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                            Int cube_id = v_part | face_part;
                            auto [is_valid, cube_value] = get_cube_validity_and_value(cube_id, negate, cell_centric);
                            if (is_valid) {
                                local_cubes.emplace_back(Cube<Int, D>(cube_id, domain()), cube_value);
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
            std::cerr << "cube_elapsed : " << cube_elapsed << "\n";
            std::cerr << "move_elapsed : " << move_elapsed << "\n";
        }

        timer.reset();
        auto fil = GridCubeFiltration(std::move(cubes), negate, n_threads, false, false);
        auto fil_elapsed = timer.elapsed_reset();
        std::cerr << "fil_elapsed : " << fil_elapsed << "\n";
        return fil;
    }

    template<typename I, typename R, size_t DD>
    friend std::ostream& operator<<(std::ostream& out, const Grid<I, R, DD>& g);

private:
    Domain domain_;
    Real* data_ {nullptr};

    void add_freudenthal_simplices_from_vertex(const GridPoint& v,
            size_t d,
            bool negate,
            const GridPointVecVec& disps,
            SimplexVec& simplices,
            CriticalVertices& critical_vertices,
            bool return_critical_vertices) const
    {
        IdxVector v_ids(d + 1, 0);

        for(auto& deltas: disps) {
            assert(deltas.size() == d + 1);

            bool is_valid_simplex = true;

            for(size_t i = 0 ; i < d + 1 ; ++i) {
                GridPoint u = add_points(v, deltas[i]);
                if (domain_.wrap())
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
                                   CriticalVertices& critical_vertices,
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

    std::pair<bool, Real> get_cube_validity_and_value(Int cube_id, bool negate, bool cell_centric) const
    {
        auto cmp = [negate](Real a, Real b) { return negate ? a > b : a < b; };

        bool is_valid = true;
        Real value = negate ? std::numeric_limits<Real>::max() : -std::numeric_limits<Real>::max();

        if (cell_centric) {
            throw std::runtime_error("Cell centric grid value is not implemented");
        } else {
            for(auto u_local : cube_private::get_cube_vertices<Int, D>(cube_id, domain())) {

                if (not contains(u_local)) {
                    is_valid = false;
                    break;
                }

                Real u_value = value_at_vertex(u_local);

                if (cmp(value, u_value))
                    value = u_value;
            }
        }
        return {is_valid, value};
    }

}; // class Grid

template<typename Int, typename Real, size_t D>
std::ostream& operator<<(std::ostream& out, const Grid<Int, Real, D>& g)
{
    out << "Grid(" << g.domain_ << ", data = " << g.data_ << ")";
    return out;
}

}
#endif //OINEUS_GRID_H