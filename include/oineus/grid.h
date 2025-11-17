#ifndef OINEUS_GRID_H
#define OINEUS_GRID_H

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "grid_domain.h"
#include "filtration.h"
#include "simplex.h"
#include "cube.h"

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
        if (top_d > dim)
            throw std::runtime_error("bad dimension, top_d = " + std::to_string(top_d) + ", dim = " + std::to_string(dim));

        GridCubeVec cubes;
        CriticalVertices dummy_vertices;

        // calculate total number of cells to allocate memory once
        size_t total_size = 0;
        for(dim_type d = 0 ; d <= top_d ; ++d) {
            total_size += get_n_cubes_in_dimension(dim, d) * size();
        }

        cubes.reserve(total_size);

        Domain global_domain = domain_;

        for(Int v = 0 ; v < size() ; ++v) {
            Int v_part = v << dim;
            for(Int face_part = 0; face_part < (1 << dim); ++face_part) {
                Int cube_id = v_part | face_part;

                auto [is_valid, cube_value] = get_cube_validity_and_value(cube_id, negate);

                if (not is_valid)
                    continue;

                cubes.emplace_back(cube_id, cube_value);

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                if (i % 100 == 0) {
                    OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
                }
#endif
            }
        }
        return GridCubeFiltration(cubes, negate, n_threads);
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

    std::pair<bool, Real> get_cube_validity_and_value(Int cube_id, bool negate)
    {
        auto cmp = [negate](Real a, Real b) { return negate ? a > b : a < b; };

        bool is_valid = true;
        Real value = -std::numeric_limits<Real>::max();

        for(auto u_local : oineus::cube_private::get_cube_vertices(cube_id)) {

            if (not contains(u_local)) {
                is_valid = false;
                break;
            }

            Real u_value = value_at_vertex(u_local);

            if (cmp(value, u_value))
                value = u_value;
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