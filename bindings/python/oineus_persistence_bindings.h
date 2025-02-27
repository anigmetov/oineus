#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <variant>
#include <functional>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//#define OINEUS_CHECK_FOR_PYTHON_INTERRUPT {if (PyErr_CheckSignals() != 0) throw py::error_already_set();}
//#define OINEUS_CHECK_FOR_PYTHON_INTERRUPT_WITH_GIL { py::gil_scoped_acquire acq; if (PyErr_CheckSignals() != 0) throw py::error_already_set(); }

#include <oineus/timer.h>
#include <oineus/oineus.h>

#ifndef OINEUS_PYTHON_INT
#define OINEUS_PYTHON_INT long int
#endif

#ifndef OINEUS_PYTHON_REAL
#define OINEUS_PYTHON_REAL double
#endif

using oin_int = OINEUS_PYTHON_INT;
using oin_real = OINEUS_PYTHON_REAL;


static_assert(std::is_same<oin_int, int>::value ||
              std::is_same<oin_int, long int>::value ||
              std::is_same<oin_int, long long int>::value,
              "OINEUS_PYTHON_INT must be one of the int, long, long long");

static_assert(std::is_same<oin_real, float>::value ||
              std::is_same<oin_real, double>::value,
              "OINEUS_PYTHON_REAL must be one of the float, double");

namespace oin = oineus;
using dim_type = oin::dim_type;

template<class Real>
class PyOineusDiagrams {
public:
    using Coordinate = Real;

    PyOineusDiagrams() = default;

    PyOineusDiagrams(const oin::Diagrams<Real>& _diagrams)
            :diagrams_(_diagrams) { }

    PyOineusDiagrams(oin::Diagrams<Real>&& _diagrams)
            :diagrams_(_diagrams) { }

    template<class R>
    py::array_t<R> diagram_to_numpy(const typename oin::Diagrams<R>::Dgm& dgm) const
    {
        size_t arr_sz = dgm.size() * 2;
        R* ptr = new R[arr_sz];
        for(size_t i = 0 ; i < dgm.size() ; ++i) {
            ptr[2 * i] = dgm[i].birth;
            ptr[2 * i + 1] = dgm[i].death;
        }

        py::capsule free_when_done(ptr, [](void* p) {
          R* pp = reinterpret_cast<R*>(p);
          delete[] pp;
        });

        py::array::ShapeContainer shape {static_cast<long int>(dgm.size()), 2L};
        py::array::StridesContainer strides {static_cast<long int>(2 * sizeof(R)),
                                             static_cast<long int>(sizeof(R))};

        return py::array_t<R>(shape, strides, ptr, free_when_done);
    }

    py::array_t<Real> get_diagram_in_dimension_as_numpy(dim_type d)
    {
        auto dgm = diagrams_.get_diagram_in_dimension(d);
        return diagram_to_numpy<Real>(dgm);
    }

    auto get_diagram_in_dimension(dim_type d)
    {
        return diagrams_.get_diagram_in_dimension(d);
    }

    auto get_index_diagram_in_dimension(dim_type d, bool sorted = true)
    {
        return diagrams_.get_index_diagram_in_dimension(d, sorted);
    }

    py::array_t<size_t> get_index_diagram_in_dimension_as_numpy(dim_type d, bool sorted = true)
    {
        auto index_dgm = diagrams_.get_index_diagram_in_dimension(d, sorted);
        return diagram_to_numpy<size_t>(index_dgm);
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
get_grid(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool wrap)
{
    using Grid = oin::Grid<Int, Real, D>;
    using GridPoint = typename Grid::GridPoint;

    py::buffer_info data_buf = data.request();

    if (data.ndim() != D)
        throw std::runtime_error("Dimension mismatch");

    Real* pdata {static_cast<Real*>(data_buf.ptr)};

    GridPoint dims;
    for(dim_type d = 0 ; d < D ; ++d)
        dims[d] = data.shape(d);

    return Grid(dims, wrap, pdata);
}

template<class Int, class Real>
decltype(auto)
get_fr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    dim_type d = data.ndim();
    if (d == 1) {
        return get_grid<Int, Real, 1>(data, wrap).freudenthal_filtration(max_dim, negate, n_threads);
    } else if (d == 2) {
        return get_grid<Int, Real, 2>(data, wrap).freudenthal_filtration(max_dim, negate, n_threads);
    } else if (d == 3) {
        return get_grid<Int, Real, 3>(data, wrap).freudenthal_filtration(max_dim, negate, n_threads);
    } else if (d == 4) {
        return get_grid<Int, Real, 4>(data, wrap).freudenthal_filtration(max_dim, negate, n_threads);
    } else {
        throw std::runtime_error("get_fr_filtration: dimension not supported by default, manual modification needed");
    }
}

template<class Int, class Real>
decltype(auto)
get_fr_filtration_and_critical_vertices(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{    dim_type d = data.ndim();
    if (d == 1) {
        return get_grid<Int, Real, 1>(data, wrap).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else if (d == 2) {
        return get_grid<Int, Real, 2>(data, wrap).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else if (d == 3) {
        return get_grid<Int, Real, 3>(data, wrap).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else if (d == 4) {
        return get_grid<Int, Real, 4>(data, wrap).freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
    } else {
        throw std::runtime_error("get_fr_filtration: dimension not supported by default, manual modification needed");
    }
}

template<class Real, size_t D>
decltype(auto) numpy_to_point_vector(py::array_t<Real, py::array::c_style | py::array::forcecast> data)
{
    using PointVector = std::vector<oin::Point<Real, D>>;

    if (data.ndim() != 2 or data.shape(1) != D)
        throw std::runtime_error("Dimension mismatch");

    py::buffer_info data_buf = data.request();

    PointVector points(data.shape(0));

    Real* pdata {static_cast<Real*>(data_buf.ptr)};

    for(ssize_t i = 0 ; i < data.size() ; ++i)
        points[i / D][i % D] = pdata[i];

    return points;
}

template<typename Int, typename Real>
decltype(auto)
list_to_filtration(py::list data) //take a list of cells and turn it into a filtration for oineus. The list should contain cells in the form '[id, [boundary], filtration value]'.
{
    using Fil = oin::Filtration<oin::Simplex<Int>, Real>;
    using Simplex = oin::Simplex<Int>;
    using SimplexVector = typename Fil::CellVector;

    int n_simps = data.size();
    SimplexVector FSV;

    for(int i = 0 ; i < n_simps ; i++) {
        auto data_i = data[i];
        int count = 0;
        Int id;
        std::vector<Int> vertices;
        Real val;
        for(auto item: data_i) {
            if (count == 0) {
                id = item.cast<Int>();
            } else if (count == 1) {
                vertices = item.cast<std::vector<Int>>();
            } else if (count == 2) {
                val = item.cast<Real>();
            }
            count++;
        }
        FSV.emplace_back(Simplex(id, vertices), val);
    }

    return Fil(FSV, false, 1);
}

template<typename Int, typename Real>
decltype(auto)
get_ls_filtration(const py::list& simplices, const py::array_t<Real>& vertex_values, bool negate, int n_threads)
// take a list of cells and a numpy array of their values and turn it into a filtration for oineus.
// The list should contain cells, each simplex is a list of vertices,
// e.g., triangulation of one segment is [[0], [1], [0, 1]]
{
    using Fil = oin::Filtration<oin::Simplex<Int>, Real>;
    using Simplex = typename Fil::Cell;
    using IdxVector = typename Simplex::Cell::IdxVector;
    using SimplexVector = std::vector<Simplex>;

    Timer timer;
    timer.reset();

    SimplexVector fil_simplices;
    fil_simplices.reserve(simplices.size());

    if (vertex_values.ndim() != 1) {
        std::cerr << "get_ls_filtration: expected 1-dimensional array in get_ls_filtration, got " << vertex_values.ndim() << std::endl;
        throw std::runtime_error("Expected 1-dimensional array in get_ls_filtration");
    }

    auto cmp = negate ? [](Real x, Real y) { return x > y; } : [](Real x, Real y) { return x < y; };

    auto vv_buf = vertex_values.request();
    Real* p_vertex_values = static_cast<Real*>(vv_buf.ptr);

    for(auto&& item: simplices) {
        IdxVector vertices = item.cast<IdxVector>();

        Real critical_value = negate ? std::numeric_limits<Real>::max() : std::numeric_limits<Real>::lowest();

        for(auto v: vertices) {
            Real vv = p_vertex_values[v];
            if (cmp(critical_value, vv)) {
                critical_value = vv;
            }
        }

        fil_simplices.emplace_back(vertices, critical_value);
    }

    return Fil(std::move(fil_simplices), negate, n_threads);
}

template<class Int, class Real>
decltype(auto)
get_vr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> points, dim_type max_dim, Real max_diameter, int n_threads)
{
     if (points.ndim() != 2)
        throw std::runtime_error("get_vr_filtration: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration<Int, Real, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration<Int, Real, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration<Int, Real, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration<Int, Real, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration<Int, Real, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("get_vr_filtration: dimension not supported by default, recompilation needed");
    }
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_and_critical_edges_from_pwdists(py::array_t<Real, py::array::c_style | py::array::forcecast> pw_dists, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (pw_dists.ndim() != 2 or pw_dists.shape(0) != pw_dists.shape(1))
        throw std::runtime_error("Dimension mismatch");

    py::buffer_info pw_dists_buf = pw_dists.request();

    Real* pdata {static_cast<Real*>(pw_dists_buf.ptr)};

    size_t n_points = pw_dists.shape(1);

    oin::DistMatrix<Real> dist_matrix {pdata, n_points};

    return oin::get_vr_filtration_and_critical_edges<Int, Real>(dist_matrix, max_dim, max_diameter, n_threads);
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_from_pwdists(py::array_t<Real, py::array::c_style | py::array::forcecast> pw_dists, dim_type max_dim, Real max_diameter, int n_threads)
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
get_vr_filtration_and_critical_edges(py::array_t<Real, py::array::c_style | py::array::forcecast> points, dim_type max_dim, Real max_diameter, int n_threads)
{
    if (points.ndim() != 2)
        throw std::runtime_error("get_vr_filtration_and_critical_edges: expected 2D array");

    dim_type d = points.shape(1);

    if (d == 1) {
        return oin::get_vr_filtration_and_critical_edges<Int, Real, 1>(numpy_to_point_vector<Real, 1>(points), max_dim, max_diameter, n_threads);
    } else if (d == 2) {
        return oin::get_vr_filtration_and_critical_edges<Int, Real, 2>(numpy_to_point_vector<Real, 2>(points), max_dim, max_diameter, n_threads);
    } else if (d == 3) {
        return oin::get_vr_filtration_and_critical_edges<Int, Real, 3>(numpy_to_point_vector<Real, 3>(points), max_dim, max_diameter, n_threads);
    } else if (d == 4) {
        return oin::get_vr_filtration_and_critical_edges<Int, Real, 4>(numpy_to_point_vector<Real, 4>(points), max_dim, max_diameter, n_threads);
    } else if (d == 5) {
        return oin::get_vr_filtration_and_critical_edges<Int, Real, 5>(numpy_to_point_vector<Real, 5>(points), max_dim, max_diameter, n_threads);
    } else {
        throw std::runtime_error("get_vr_filtration_and_critical_edges: dimension " + std::to_string(d) + " not supported by default, recompilation needed");
    }
}

template<class Int, class Real>
typename oin::VRUDecomposition<Int>::MatrixData
get_boundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real>(data, negate, wrap, max_dim, n_threads);
    return fil.boundary_matrix();
}

template<class Int, class Real, size_t D>
typename oin::VRUDecomposition<Int>::MatrixData
get_coboundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads);
    auto bm = fil.boundary_matrix();
    return oin::antitranspose(bm);
}

template<class Int, class Real>
PyOineusDiagrams<Real>
compute_diagrams_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, oin::Params& params, bool include_inf_points, bool dualize)
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


void init_oineus_common(py::module& m);
void init_oineus_common_decomposition(py::module& m);
void init_oineus_diagram(py::module& m);
void init_oineus_functions(py::module& m);
void init_oineus_filtration(py::module& m);
void init_oineus_cells(py::module& m);
void init_oineus_kicr(py::module& m);
void init_oineus_top_optimizer(py::module& m);

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
