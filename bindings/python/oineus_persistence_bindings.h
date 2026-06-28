#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#define OINEUS_PYTHON_FRIENDS

#include <iostream>
#include <vector>
#include <sstream>
#include <variant>
#include <functional>
#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
// SBO column caster: lets boost::container::small_vector<T,N> (the at-rest
// reduction-column type) round-trip as a Python list, exactly like
// std::vector<T>. This keeps r_data/v_data/u_data_t exposed as list-of-lists
// (def_rw, pickle, the MatrixData ctor, diagram/boundary returns) with no
// other binding changes. Harmless under the OINEUS_COL_USE_STD_VECTOR
// baseline -- the specialization is simply never instantiated then.
#include <boost/container/small_vector.hpp>
namespace nanobind { namespace detail {
template <typename T, std::size_t N, typename A>
struct type_caster<boost::container::small_vector<T, N, A>>
    : list_caster<boost::container::small_vector<T, N, A>, T> {};
}}
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include "nanobind/stl/tuple.h"
#include <nanobind/nb_python.h>
#include <nanobind/ndarray.h>
#include "nanobind/operators.h"
#include "nanobind/make_iterator.h"

#include "uid128_caster.h"
#include "oineus_signal_guard.h"

namespace nb = nanobind;

using namespace nb::literals;


#include <oineus/timer.h>
#include <oineus/oineus.h>

#ifndef OINEUS_COL_USE_STD_VECTOR
// The at-rest column is oineus::OinColumn (derives from small_vector). It is a
// distinct type from its base, so the small_vector caster above does not apply --
// give it its own list_caster so r_data/v_data/u_data_t still round-trip as Python
// lists of lists. Defined after oineus.h, where OinColumn is complete.
namespace nanobind { namespace detail {
template <typename Int>
struct type_caster<oineus::OinColumn<Int>>
    : list_caster<oineus::OinColumn<Int>, Int> {};
}}
#endif

// jemalloc-backed std::unordered_map (Filtration::uid_to_sorted_id, which is part
// of the pickle state tuple). nanobind's built-in unordered_map caster is a 4-arg
// partial specialization that only matches the DEFAULT allocator, so the
// JeAllocator variant needs its own. Specializing on JeAllocator (rather than a
// generic 5th allocator param) avoids ambiguity with the built-in caster for
// default-allocator maps.
namespace nanobind { namespace detail {
template <typename Key, typename T, typename Hash, typename Eq, typename A>
struct type_caster<std::unordered_map<Key, T, Hash, Eq, oineus::JeAllocator<A>>>
    : dict_caster<std::unordered_map<Key, T, Hash, Eq, oineus::JeAllocator<A>>, Key, T> {};
}}

#ifndef OINEUS_PYTHON_INT
#define OINEUS_PYTHON_INT long int
#endif

#ifndef OINEUS_PYTHON_REAL
#define OINEUS_PYTHON_REAL double
#endif

using oin_int = OINEUS_PYTHON_INT;
using oin_real = OINEUS_PYTHON_REAL;




// using Z2_Column = oineus::SimpleSparseMatrixTraits<oin_int, 2>::Column;
// using Z2_Matrix = oineus::SimpleSparseMatrixTraits<oin_int, 2>::Matrix;
// PYBIND11_MAKE_OPAQUE(Z2_Column);
// PYBIND11_MAKE_OPAQUE(Z2_Matrix);


static_assert(std::is_same<oin_int, int>::value ||
              std::is_same<oin_int, long int>::value ||
              std::is_same<oin_int, long long int>::value,
              "OINEUS_PYTHON_INT must be one of the int, long, long long");

static_assert(std::is_same<oin_real, float>::value ||
              std::is_same<oin_real, double>::value,
              "OINEUS_PYTHON_REAL must be one of the float, double");

namespace oin = oineus;
using dim_type = oin::dim_type;

// The cube filtration stores slim cubes (uid + user_id only); the shared
// GridDomain is owned once by the filtration. Python, however, expects a
// self-contained "fat" cube (carrying its domain) so it can call boundary(),
// vertices, etc. without a filtration in hand. These helpers convert between the
// stored slim form and the materialized fat form at the binding boundary: the
// filtration accessors return fat cubes (fatten_cube + the filtration's
// geometry()), and the filtration constructor-from-cells slims them back
// (slim_cube), recovering the shared domain from the cells.
template<class Int, unsigned D, class Real>
oin::CellWithValue<oin::FatCube<Int, D>, Real>
fatten_cube(const oin::CellWithValue<oin::Cube<Int, D>, Real>& slim, const oin::GridDomain<Int, D>& dom)
{
    oin::CellWithValue<oin::FatCube<Int, D>, Real> fat(oin::FatCube<Int, D>(slim.get_cell(), dom), slim.get_value());
    fat.set_sorted_id(slim.get_sorted_id());
    return fat;
}

template<class Int, unsigned D, class Real>
oin::CellWithValue<oin::Cube<Int, D>, Real>
slim_cube(const oin::CellWithValue<oin::FatCube<Int, D>, Real>& fat)
{
    oin::Cube<Int, D> c(fat.get_cell().get_uid());
    c.set_id(fat.get_cell().get_id());
    oin::CellWithValue<oin::Cube<Int, D>, Real> slim(c, fat.get_value());
    slim.set_sorted_id(fat.get_sorted_id());
    return slim;
}

// Materialize every stored slim cube into its fat Python form (one shared
// geometry copy per cell). Used by the cube-filtration accessors that hand the
// whole cell sequence to Python (cells/cubes/__iter__/__getstate__).
template<class Int, unsigned D, class Real>
std::vector<oin::CellWithValue<oin::FatCube<Int, D>, Real>>
fatten_all(const oin::Filtration<oin::Cube<Int, D>, Real>& fil)
{
    std::vector<oin::CellWithValue<oin::FatCube<Int, D>, Real>> fat;
    fat.reserve(fil.size());
    for (const auto& cv : fil.cells())
        fat.push_back(fatten_cube<Int, D, Real>(cv, fil.geometry()));
    return fat;
}

// The slim Freudenthal filtration stores compact (anchor,type) cells
// (Simplex<Int, FreudenthalAnchorType>) + one shared FrGeometry. As with cubes,
// Python expects a self-contained fat cell -- here a fat Simplex<Int>, because a
// Freudenthal cell IS the simplex on its grid vertices (FrGeometry::vertices_of).
// These helpers convert at the binding boundary: filtration accessors materialize fat
// simplices (fatten_simplex_from_fr + the filtration's geometry()), and the
// constructor/unpickle slim them back (slim_simplex_from_fr). The pickle stores the
// GridDomain (a bound type) rather than the unbound FrGeometry and rebuilds the
// geometry on restore.
template<class Int, unsigned D, class Real>
oin::CellWithValue<oin::Simplex<Int>, Real>
fatten_simplex_from_fr(const oin::CellWithValue<oin::Simplex<Int, oin::FreudenthalAnchorType<Int, D>>, Real>& slim,
                       const oin::FrGeometry<Int, D>& geom)
{
    auto vids = slim.get_cell().vertices(geom);  // sorted fat grid-vertex ids
    typename oin::Simplex<Int>::IdxVector iv(vids.begin(), vids.end());
    oin::Simplex<Int> sigma(slim.get_cell().get_id(), iv);
    oin::CellWithValue<oin::Simplex<Int>, Real> fat(std::move(sigma), slim.get_value());
    fat.set_sorted_id(slim.get_sorted_id());
    return fat;
}

template<class Int, unsigned D, class Real>
oin::CellWithValue<oin::Simplex<Int, oin::FreudenthalAnchorType<Int, D>>, Real>
slim_simplex_from_fr(const oin::CellWithValue<oin::Simplex<Int>, Real>& fat, const oin::FrGeometry<Int, D>& geom)
{
    std::vector<Int> vids(fat.get_cell().get_vertices().begin(), fat.get_cell().get_vertices().end());
    Int uid = geom.uid_of_vertices(vids);
    oin::Simplex<Int, oin::FreudenthalAnchorType<Int, D>> slim_cell(
            oin::FreudenthalAnchorType<Int, D>(uid, geom.dim_of_uid(uid)), fat.get_cell().get_id());
    oin::CellWithValue<oin::Simplex<Int, oin::FreudenthalAnchorType<Int, D>>, Real> slim(slim_cell, fat.get_value());
    slim.set_sorted_id(fat.get_sorted_id());
    return slim;
}

template<class Int, unsigned D, class Real>
std::vector<oin::CellWithValue<oin::Simplex<Int>, Real>>
fatten_all_fr(const oin::Filtration<oin::Simplex<Int, oin::FreudenthalAnchorType<Int, D>>, Real>& fil)
{
    std::vector<oin::CellWithValue<oin::Simplex<Int>, Real>> fat;
    fat.reserve(fil.size());
    for (const auto& cv : fil.cells())
        fat.push_back(fatten_simplex_from_fr<Int, D, Real>(cv, fil.geometry()));
    return fat;
}

// Bit-packed VR/alpha filtrations: the same materialize-to-fat-Simplex pattern as the
// slim Freudenthal one (a packed simplex IS the simplex on its unpacked vertices), but
// the geometry is a trivially-copyable PackedGeom{int bits}, so pickle stores the bits
// int directly (no unbound table-bearing geometry to rebuild). Shared by VR and alpha,
// which both materialize to the universal fat Simplex<Int>. Templated on the word width
// Word so one definition serves both packed tiers (uint64 / unsigned __int128).
template<class Int, class Word, class Real>
oin::CellWithValue<oin::Simplex<Int>, Real>
fatten_simplex_from_packed(const oin::CellWithValue<oin::Simplex<Int, oin::BitPacked<Int, Word>>, Real>& slim,
                           const oin::PackedGeom& geom)
{
    auto vids = slim.get_cell().vertices(geom);  // ascending unpacked vertex ids
    typename oin::Simplex<Int>::IdxVector iv(vids.begin(), vids.end());
    oin::Simplex<Int> sigma(slim.get_cell().get_id(), iv);
    oin::CellWithValue<oin::Simplex<Int>, Real> fat(std::move(sigma), slim.get_value());
    fat.set_sorted_id(slim.get_sorted_id());
    return fat;
}

template<class Int, class Word, class Real>
oin::CellWithValue<oin::Simplex<Int, oin::BitPacked<Int, Word>>, Real>
slim_simplex_from_packed(const oin::CellWithValue<oin::Simplex<Int>, Real>& fat, const oin::PackedGeom& geom)
{
    // fat Simplex vertices are already ascending, so the BitPacked ctor packs directly
    const auto& vids = fat.get_cell().get_vertices();
    oin::Simplex<Int, oin::BitPacked<Int, Word>> slim_cell(
            oin::BitPacked<Int, Word>(vids, geom.bits), fat.get_cell().get_id());
    oin::CellWithValue<oin::Simplex<Int, oin::BitPacked<Int, Word>>, Real> slim(slim_cell, fat.get_value());
    slim.set_sorted_id(fat.get_sorted_id());
    return slim;
}

template<class Int, class Word, class Real>
std::vector<oin::CellWithValue<oin::Simplex<Int>, Real>>
fatten_all_packed(const oin::Filtration<oin::Simplex<Int, oin::BitPacked<Int, Word>>, Real>& fil)
{
    std::vector<oin::CellWithValue<oin::Simplex<Int>, Real>> fat;
    fat.reserve(fil.size());
    for (const auto& cv : fil.cells())
        fat.push_back(fatten_simplex_from_packed<Int, Word, Real>(cv, fil.geometry()));
    return fat;
}

// uid-contract translation for the Python-facing uid accessors of slim/packed
// filtrations. A Python cell always carries the universal COMBINATORIAL uid (a function
// of its sorted vertex set, identical across any filtration containing that simplex),
// but a slim Freudenthal / bit-packed filtration is keyed internally by its own compact
// uid. These helpers decode the combinatorial uid back to its vertex set
// (vertices_from_simplex_uid) and re-key it into the filtration's internal form, so
// value_by_uid / sorted_id_by_uid / cell_by_uid accept the same uid the user reads off a
// cell. Both are pure (no shared mutable state) -- safe to call concurrently. A uid whose
// vertex set is not a cell of this filtration is reported as "not present" (out_of_range),
// matching the fat Simplex accessor contract.
template<class Int, unsigned D>
Int fr_slim_uid_from_comb_uid(const oin::FrGeometry<Int, D>& geom, unsigned __int128 comb_uid)
{
    std::vector<Int> vids = oin::vertices_from_simplex_uid<Int>(comb_uid);
    try {
        return geom.uid_of_vertices(vids);
    } catch (const std::runtime_error&) {
        // vertex set is not a Freudenthal simplex of this grid -> not a cell here
        throw std::out_of_range("Filtration: uid not present in filtration");
    }
}

template<class Int, class Word>
Word packed_word_uid_from_comb_uid(const oin::PackedGeom& geom, unsigned __int128 comb_uid)
{
    std::vector<Int> vids = oin::vertices_from_simplex_uid<Int>(comb_uid);
    const int field_bits = geom.bits;
    const int word_bits = static_cast<int>(8 * sizeof(Word));
    // A foreign uid may decode to a simplex that does not FIT this filtration's packing:
    // too many vertices for the Word, or a vertex id wider than the field. BitPacked::pack
    // would then bit-spill (e.g. pack({32}) == pack({0,1}) at bits=5) and the Word could
    // alias a different, present cell -- silently returning the wrong cell. Such a simplex
    // is not in this filtration, so report "not present" (matching the fat accessor).
    if (static_cast<int>(vids.size()) * field_bits > word_bits)
        throw std::out_of_range("Filtration: uid not present in filtration");
    if (field_bits < word_bits) {
        for (Int v : vids)
            if (v < 0 || (static_cast<Word>(v) >> field_bits) != 0)
                throw std::out_of_range("Filtration: uid not present in filtration");
    }
    return oin::BitPacked<Int, Word>::pack(vids, geom.bits);
}

template<class Real>
class PyOineusDiagrams {
public:
    using Coordinate = Real;
    using Storage = oin::Diagrams<Real>;
    using State = std::vector<typename Storage::Dgm>;

    PyOineusDiagrams() = default;

    explicit PyOineusDiagrams(dim_type max_dim)
            :diagrams_(max_dim) { }

    PyOineusDiagrams(const oin::Diagrams<Real>& _diagrams)
            :diagrams_(_diagrams) { }

    PyOineusDiagrams(oin::Diagrams<Real>&& _diagrams)
            :diagrams_(_diagrams) { }

    template<class R>
    nb::ndarray<R, nb::numpy> diagram_to_numpy(const typename oin::Diagrams<R>::Dgm& dgm) const
    {
        size_t arr_sz = dgm.size() * 2;
        R* ptr = new R[arr_sz];
        for(size_t i = 0 ; i < dgm.size() ; ++i) {
            ptr[2 * i] = dgm[i].birth;
            ptr[2 * i + 1] = dgm[i].death;
        }

        nb::capsule free_when_done(ptr, [](void* p) noexcept {
          R* pp = reinterpret_cast<R*>(p);
          delete[] pp;
        });

        return nb::ndarray<R, nb::numpy>(ptr, {dgm.size(), static_cast<size_t>(2)}, free_when_done);
    }

    nb::ndarray<Real, nb::numpy> get_diagram_in_dimension_as_numpy(dim_type d)
    {
        auto dgm = diagrams_.get_diagram_in_dimension(d);
        return diagram_to_numpy<Real>(dgm);
    }

    auto get_diagram_in_dimension(dim_type d)
    {
        return diagrams_.get_diagram_in_dimension(d);
    }

    auto get_index_diagram_in_dimension(dim_type d)
    {
        return diagrams_.get_index_diagram_in_dimension(d);
    }

    nb::ndarray<size_t, nb::numpy> get_index_diagram_in_dimension_as_numpy(dim_type d)
    {
        auto index_dgm = diagrams_.get_index_diagram_in_dimension(d);
        return diagram_to_numpy<size_t>(index_dgm);
    }

    size_t n_dims() const
    {
        return diagrams_.n_dims();
    }

    void pad_to_dim(dim_type new_top_dim)
    {
        diagrams_.pad_to_dim(new_top_dim);
    }

    void trim_to_dim(dim_type max_dim)
    {
        diagrams_.trim_to_dim(max_dim);
    }

    const Storage& data() const
    {
        return diagrams_;
    }

    void set_data(const Storage& diagrams)
    {
        diagrams_ = diagrams;
    }

    State state() const
    {
        return diagrams_.diagram_in_dimension_;
    }

    void set_state(const State& state)
    {
        diagrams_.diagram_in_dimension_ = state;
    }

    bool operator==(const PyOineusDiagrams& other) const
    {
        return diagrams_ == other.diagrams_;
    }

    bool operator!=(const PyOineusDiagrams& other) const
    {
        return !(*this == other);
    }

private:
    oin::Diagrams<Real> diagrams_;
};

template<class Int, class Real>
using DiagramV = std::pair<PyOineusDiagrams<Real>, typename oin::VRUDecomposition<Int>::MatrixData>;

template<class Int, class Real>
using DiagramRV = std::tuple<PyOineusDiagrams<Real>, typename oin::VRUDecomposition<Int>::MatrixData, typename oin::VRUDecomposition<Int>::MatrixData>;

template<class Int, class Real, size_t D>
typename oin::Grid<Int, Real, D>
get_grid(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data, bool wrap, bool cell_centric)
{
    using Grid = oin::Grid<Int, Real, D>;
    using GridPoint = typename Grid::GridPoint;

    if (data.ndim() != D)
        throw std::runtime_error("get_grid: expected array of dimension " + std::to_string(D));

    const Real* const pdata {static_cast<const Real* const>(data.data())};

    GridPoint dims;
    for(dim_type d = 0 ; d < D ; ++d)
        dims[d] = data.shape(d);

    typename Grid::DataLocation data_location = cell_centric ? Grid::DataLocation::CELL : Grid::DataLocation::VERTEX;

    return Grid(dims, wrap, pdata, data_location);
}

template<class Int, class Real>
decltype(auto)
get_fr_filtration(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    dim_type d = data.ndim();
    if (d == 1) {
        return get_grid<Int, Real, 1>(data, wrap, false).freudenthal_filtration(max_dim, negate, n_threads);
    } else if (d == 2) {
        return get_grid<Int, Real, 2>(data, wrap, false).freudenthal_filtration(max_dim, negate, n_threads);
    } else if (d == 3) {
        return get_grid<Int, Real, 3>(data, wrap, false).freudenthal_filtration(max_dim, negate, n_threads);
    } else if (d == 4) {
        return get_grid<Int, Real, 4>(data, wrap, false).freudenthal_filtration(max_dim, negate, n_threads);
    } else {
        throw std::runtime_error("get_fr_filtration: dimension not supported by default, manual modification needed");
    }
}

template<class Int, class Real>
decltype(auto)
get_fr_filtration_and_critical_vertices(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    dim_type d = data.ndim();
    if (d == 1) {
        return get_grid<Int, Real, 1>(data, wrap, false).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else if (d == 2) {
        return get_grid<Int, Real, 2>(data, wrap, false).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else if (d == 3) {
        return get_grid<Int, Real, 3>(data, wrap, false).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else if (d == 4) {
        return get_grid<Int, Real, 4>(data, wrap, false).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else {
        throw std::runtime_error("get_fr_filtration: dimension not supported by default, manual modification needed");
    }
}


template<class Real, size_t D>
decltype(auto) numpy_to_point_vector(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data)
{
    if (data.ndim() != 2)
        throw std::runtime_error("numpy_to_point_vector: expected array with 2 dimensions");

    if (data.shape(1) != D)
        throw std::runtime_error("numpy_to_point_vector: expected array with dimension 1 shape = " + std::to_string(D) + ", got " + std::to_string(data.shape(1)));

    using PointVector = std::vector<oin::Point<Real, D>>;

    PointVector points(data.shape(0));

    const Real* pdata {static_cast<const Real*>(data.data())};

    for(size_t i = 0 ; i < data.size() ; ++i)
        points[i / D][i % D] = pdata[i];

    return points;
}


// template<typename Int, typename Real>
// decltype(auto)
// get_ls_filtration(const nb::list& simplices, const nb::ndarray<Real, nb::device::cpu, nb::c_contig, nb::ro>& vertex_values, bool negate, int n_threads)
// // take a list of cells and a numpy array of their values and turn it into a filtration for oineus.
// // The list should contain cells, each simplex is a list of vertices,
// // e.g., triangulation of one segment is [[0], [1], [0, 1]]
// {
//     using Fil = oin::Filtration<oin::Simplex<Int>, Real>;
//     using Simplex = typename Fil::Cell;
//     using IdxVector = typename Simplex::Cell::IdxVector;
//     using SimplexVector = std::vector<Simplex>;
//
//     Timer timer;
//     timer.reset();
//
//     SimplexVector fil_simplices;
//     fil_simplices.reserve(simplices.size());
//
//     if (vertex_values.ndim() != 1) {
//         std::cerr << "get_ls_filtration: expected 1-dimensional array in get_ls_filtration, got " << vertex_values.ndim() << std::endl;
//         throw std::runtime_error("Expected 1-dimensional array in get_ls_filtration");
//     }
//
//     auto cmp = negate ? [](Real x, Real y) { return x > y; } : [](Real x, Real y) { return x < y; };
//
//     auto vv_buf = vertex_values.request();
//     Real* p_vertex_values = static_cast<Real*>(vv_buf.ptr);
//
//     for(auto&& item: simplices) {
//         IdxVector vertices = item.cast<IdxVector>();
//
//         Real critical_value = negate ? std::numeric_limits<Real>::max() : std::numeric_limits<Real>::lowest();
//
//         for(auto v: vertices) {
//             Real vv = p_vertex_values[v];
//             if (cmp(critical_value, vv)) {
//                 critical_value = vv;
//             }
//         }
//
//         fil_simplices.emplace_back(vertices, critical_value);
//     }
//
//     return Fil(std::move(fil_simplices), negate, n_threads);
// }

// Vietoris-Rips bindings: implemented via the in-order (VRE) algorithm of
// Vejdemo-Johansson, Matuszewski & Bauer (arXiv:2411.05495). The legacy
// Bron-Kerbosch implementation remains in the C++ headers for use as a
// reference and for benchmarking, but is not exposed to Python.

template<class Int, class Real>
decltype(auto)
get_vr_filtration(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> points, dim_type max_dim, Real max_diameter, int n_threads)
{
     if (points.ndim() != 2)
        throw std::runtime_error("get_vr_filtration: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration_inorder<Int, Real, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration_inorder<Int, Real, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration_inorder<Int, Real, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration_inorder<Int, Real, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration_inorder<Int, Real, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("get_vr_filtration: dimension not supported by default, recompilation needed");
    }
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_and_critical_edges_from_pwdists(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> pw_dists, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (pw_dists.ndim() != 2 or pw_dists.shape(0) != pw_dists.shape(1))
        throw std::runtime_error("Dimension mismatch");

    const Real* pdata {static_cast<const Real*>(pw_dists.data())};

    size_t n_points = pw_dists.shape(1);

    oin::DistMatrix<Real> dist_matrix {pdata, n_points};

    return oin::get_vr_filtration_and_critical_edges_inorder<Int, Real>(dist_matrix, max_dim, max_diameter, n_threads);
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_from_pwdists(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> pw_dists, dim_type max_dim, Real max_diameter, int n_threads)
{
    return get_vr_filtration_and_critical_edges_from_pwdists<Int, Real>(pw_dists, max_dim, max_diameter, n_threads).first;
}

template<class Cell, class Real>
PyOineusDiagrams<Real>
compute_diagrams_from_fil(const oineus::Filtration<Cell, Real>& fil, int n_threads)
{
    using Int = typename Cell::Int;
    oineus::VRUDecomposition<Int> d_matrix {fil, false};

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil));
}

template<class Cell, class Real>
PyOineusDiagrams<Real>
compute_relative_diagrams(const oineus::Filtration<Cell, Real>& fil, const oineus::Filtration<Cell, Real>& relative, bool include_inf_points)
{
    using Int = typename Cell::Int;

    typename Cell::UidSet relative_;
    for(const auto& sigma : relative.cells()) {
        relative_.insert(sigma.get_uid());
    }

    auto rel_matrix = fil.boundary_matrix_rel(relative_);
    oineus::VRUDecomposition<Int> d_matrix {rel_matrix, false};

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = 1;

    d_matrix.reduce(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil, relative_, include_inf_points));
}

template<class Int, class Real>
decltype(auto)
get_vr_filtration_and_critical_edges(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> points, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (points.ndim() != 2)
        throw std::runtime_error("get_vr_filtration_and_critical_edges: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration_and_critical_edges_inorder<Int, Real, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration_and_critical_edges_inorder<Int, Real, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration_and_critical_edges_inorder<Int, Real, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration_and_critical_edges_inorder<Int, Real, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration_and_critical_edges_inorder<Int, Real, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("get_vr_filtration_and_critical_edges: dimension " + std::to_string(d) + " not supported by default, recompilation needed");
    }
}

// Bit-packed VR builders: same dispatch on the spatial dimension as the fat versions
// above, but emit Filtration<Simplex<Int,BitPacked<Int,Word>>>. Templated on the word
// width Word (the Python factory picks it via bit_packing_fits before calling, so the
// vertex ids are guaranteed to fit). All spatial-dimension branches return the same
// packed Filtration type, so decltype(auto) is single-typed as in the fat versions.
template<class Int, class Real, class Word>
decltype(auto)
get_vr_filtration_packed(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> points, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (points.ndim() != 2)
        throw std::runtime_error("get_vr_filtration_packed: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration_packed_inorder<Int, Real, Word, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration_packed_inorder<Int, Real, Word, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration_packed_inorder<Int, Real, Word, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration_packed_inorder<Int, Real, Word, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration_packed_inorder<Int, Real, Word, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("get_vr_filtration_packed: dimension not supported by default, recompilation needed");
    }
}

template<class Int, class Real, class Word>
decltype(auto)
get_vr_filtration_and_critical_edges_packed(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> points, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (points.ndim() != 2)
        throw std::runtime_error("get_vr_filtration_and_critical_edges_packed: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, Word, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, Word, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, Word, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, Word, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, Word, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("get_vr_filtration_and_critical_edges_packed: dimension " + std::to_string(d) + " not supported by default, recompilation needed");
    }
}

template<class Int, class Real, class Word>
decltype(auto)
get_vr_filtration_and_critical_edges_packed_from_pwdists(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> pw_dists, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (pw_dists.ndim() != 2 or pw_dists.shape(0) != pw_dists.shape(1))
        throw std::runtime_error("Dimension mismatch");

    const Real* pdata {static_cast<const Real*>(pw_dists.data())};
    size_t n_points = pw_dists.shape(1);
    oin::DistMatrix<Real> dist_matrix {pdata, n_points};

    return oin::get_vr_filtration_and_critical_edges_packed_inorder<Int, Real, Word>(dist_matrix, max_dim, max_diameter, n_threads);
}

template<class Int, class Real, class Word>
decltype(auto)
get_vr_filtration_packed_from_pwdists(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> pw_dists, dim_type max_dim, Real max_diameter, int n_threads)
{
    return get_vr_filtration_and_critical_edges_packed_from_pwdists<Int, Real, Word>(pw_dists, max_dim, max_diameter, n_threads).first;
}

// Internal/test-only: brute-force VR construction (uses C++ naive fallback).
// Limited to max_dim <= 3 by the C++ implementation. Exposed so the test
// suite can cross-check VRE against a wholly different code path; not part
// of the user API.
template<class Int, class Real>
decltype(auto)
_get_vr_filtration_naive(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> points, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (points.ndim() != 2)
        throw std::runtime_error("_get_vr_filtration_naive: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration_naive<Int, Real, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration_naive<Int, Real, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration_naive<Int, Real, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration_naive<Int, Real, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration_naive<Int, Real, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("_get_vr_filtration_naive: dimension not supported by default, recompilation needed");
    }
}

template<class Int, class Real>
typename oin::VRUDecomposition<Int>::MatrixData
get_boundary_matrix(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real>(data, negate, wrap, max_dim, n_threads);
    return fil.boundary_matrix();
}

template<class Int, class Real, size_t D>
typename oin::VRUDecomposition<Int>::MatrixData
get_coboundary_matrix(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads);
    auto bm = fil.boundary_matrix();
    return oin::antitranspose(bm);
}

template<class Int, class Real>
PyOineusDiagrams<Real>
compute_diagrams_ls_freudenthal(nb::ndarray<Real, nb::c_contig, nb::device::cpu, nb::ro> data, bool negate, bool wrap, dim_type max_dim, oin::Params& params, bool include_inf_points, bool dualize)
{
    // for diagram in dimension d, we need (d+1)-cells
    Timer timer;
    auto fil = get_fr_filtration<Int, Real>(data, negate, wrap, max_dim + 1, params.n_threads);
    auto elapsed_fil = timer.elapsed_reset();
    oin::VRUDecomposition<Int> decmp {fil, dualize};
    auto elapsed_decmp_ctor = timer.elapsed_reset();

    if (params.print_time)
        std::cerr << "Filtration: " << elapsed_fil << ", decomposition ctor: " << elapsed_decmp_ctor << std::endl;

    decmp.reduce(params);

    if (params.do_sanity_check and not decmp.sanity_check())
        throw std::runtime_error("sanity check failed");
    return PyOineusDiagrams<Real>(decmp.diagram(fil, include_inf_points));
}


template<typename C, typename Real>
oin::KerImCokReduced<C, Real, 2> compute_kernel_image_cokernel_reduction(const oin::Filtration<C, Real>& K, const oin::Filtration<C, Real>& L, oin::Params& params)
{
    using KICR = oin::KerImCokReduced<C, Real, 2>;

    params.sort_dgms = false;
    params.clearing_opt = false;

    oin::KICRParams kicr_params;
    kicr_params.verbose = params.verbose;
    kicr_params.kernel = kicr_params.image = kicr_params.cokernel = true;
    kicr_params.params_f = kicr_params.params_g = params;
    kicr_params.params_ker = kicr_params.params_cok = kicr_params.params_im = params;

    KICR result { K, L, kicr_params };

    return result;
}


// Real-INDEPENDENT registration (enums, params, the int-templated decomposition
// stats): registered once on the top module.
void init_oineus_common(nb::module_& m);
void init_oineus_dcmp_stats(nb::module_& m);

// Real-DEPENDENT registration, templated on Real and registered per dtype. The
// double pass targets the top module (so the existing API is byte-identical) with
// reg_indep=true (it also registers the few Real-INDEPENDENT classes that live
// alongside, e.g. CombinatorialSimplex / GridDomain / IndexDiagramPoint); the float
// pass targets the _f32 submodule with reg_indep=false (those shared types are found
// in nanobind's global registry). The thin init_oineus_* wrappers below drive the
// double pass so the module driver and the per-file refactors can land incrementally.
template<class Real> void register_oineus_cells(nb::module_& m, bool reg_indep);
template<class Real> void register_oineus_diagram(nb::module_& m, bool reg_indep);
template<class Real> void register_oineus_functions(nb::module_& m, bool reg_indep);
template<class Real> void register_oineus_filtration(nb::module_& m, bool reg_indep);
template<class Real> void register_oineus_kicr(nb::module_& m, bool reg_indep);
template<class Real> void register_oineus_top_optimizer(nb::module_& m, bool reg_indep);
// Decomposition: the class (VRUDecomposition<Int>) is Real-independent but has
// Real-dependent ctors/diagram/reduce. One templated registrar handles both: the
// class + its Real-independent methods register once (reg_indep, on the top module
// via a file-scope handle shared across the passes), and every pass adds the
// Real-dependent ctor/diagram overloads to that one class plus its per-Real free
// `reduce` functions on the target module.
template<class Real> void register_oineus_decomposition(nb::module_& m, bool reg_indep);

void init_oineus_common_decomposition(nb::module_& m);
void init_oineus_diagram(nb::module_& m);
void init_oineus_functions(nb::module_& m);
void init_oineus_filtration(nb::module_& m);
void init_oineus_cells(nb::module_& m);
void init_oineus_kicr(nb::module_& m);
void init_oineus_top_optimizer(nb::module_& m);

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
