#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <oineus/oineus.h>

using dim_type = oineus::dim_type;

template<class Real>
class PyOineusDiagrams {
public:
    PyOineusDiagrams() = default;

    PyOineusDiagrams(const oineus::Diagrams<Real>& _diagrams)
            :diagrams_(_diagrams) { }

    py::array_t<Real> get_diagram_in_dimension(dim_type d)
    {
        auto dgm = diagrams_.get_diagram_in_dimension(d);

        size_t arr_sz = dgm.size() * 2;
        Real* ptr = new Real[arr_sz];
        for(size_t i = 0; i < dgm.size(); ++i) {
            ptr[2 * i] = dgm[i].birth;
            ptr[2 * i + 1] = dgm[i].death;
        }

        py::capsule free_when_done(ptr, [](void* p) {
          Real* pp = reinterpret_cast<Real*>(p);
          delete[] pp;
        });

        py::array::ShapeContainer shape {static_cast<long int>(dgm.size()), 2L};
        py::array::StridesContainer strides {static_cast<long int>(2 * sizeof(Real)),
                                             static_cast<long int>(sizeof(Real))};

        return py::array_t<Real>(shape, strides, ptr, free_when_done);
    }

private:
    oineus::Diagrams<Real> diagrams_;
};

template<class Int, class Real>
using DiagramV = std::pair<PyOineusDiagrams<Real>, typename oineus::SparseMatrix<Int>::MatrixData>;

template<class Int, class Real, size_t D>
typename oineus::Grid<Int, Real, D>
get_grid(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool wrap)
{
    using Grid = oineus::Grid<Int, Real, D>;
    using GridPoint = typename Grid::GridPoint;

    py::buffer_info data_buf = data.request();

    if (data.ndim() != D)
        throw std::runtime_error("Dimension mismatch");

    Real* pdata {static_cast<Real*>(data_buf.ptr)};

    GridPoint dims;
    for(dim_type d = 0; d < D; ++d)
        dims[d] = data.shape(d);

    return Grid(dims, wrap, pdata);
}

template<class Int, class Real, size_t D>
typename oineus::Filtration<Int, Real>
get_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto grid = get_grid<Int, Real, D>(data, wrap);
    auto res = grid.freudenthal_filtration(top_d, negate, n_threads);
    return res;
}

template<class Int, class Real, size_t D>
typename oineus::SparseMatrix<Int>::MatrixData
get_boundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
    auto bm = fil.boundary_matrix_full();
    return bm.data;
}

template<class Int, class Real, size_t D>
DiagramV<Int, Real>
compute_diagrams_and_v_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
    auto d_matrix = fil.boundary_matrix_full();

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return {PyOineusDiagrams<Real>(d_matrix.diagram(fil)), d_matrix.v_data};
}

template<class Int, class Real, size_t D>
PyOineusDiagrams<Real>
compute_diagrams_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
    auto d_matrix = fil.boundary_matrix_full();

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil));
}

template<class Int, class Real>
void init_oineus(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using RealDgmPoint = oineus::DgmPoint<Real>;
    using RealDiagram = PyOineusDiagrams<Real>;

    std::string dgm_point_name = "DiagramPoint" + suffix;
    std::string dgm_class_name = "Diagrams" + suffix;

    py::class_<RealDgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<Real, Real>())
            .def_readwrite("birth", &RealDgmPoint::birth)
            .def_readwrite("death", &RealDgmPoint::death)
            .def("__getitem__", [](const RealDgmPoint& p, int i) {
              if (i == 0)
                  return p.birth;
              else if (i == 1)
                  return p.death;
              else
                  throw std::out_of_range("i must be 0 or 1");
            })
            .def("__repr__", [](const RealDgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<RealDiagram>(m, dgm_class_name.c_str())
            .def(py::init<>())
            .def("in_dimension", &RealDiagram::get_diagram_in_dimension)
            .def("__getitem__", &RealDiagram::get_diagram_in_dimension);

    std::string func_name;

    // diagrams
    func_name = "compute_diagrams_ls" + suffix + "_1";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<Int, Real, 1>);

    func_name = "compute_diagrams_ls" + suffix + "_2";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<Int, Real, 2>);

    func_name = "compute_diagrams_ls" + suffix + "_3";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<Int, Real, 3>);

    // diagrams and V matrix
    func_name = "compute_diagrams_and_v_ls" + suffix + "_1";
    m.def(func_name.c_str(), &compute_diagrams_and_v_ls_freudenthal<Int, Real, 1>);

    func_name = "compute_diagrams_and_v_ls" + suffix + "_2";
    m.def(func_name.c_str(), &compute_diagrams_and_v_ls_freudenthal<Int, Real, 2>);

    func_name = "compute_diagrams_and_v_ls" + suffix + "_3";
    m.def(func_name.c_str(), &compute_diagrams_and_v_ls_freudenthal<Int, Real, 3>);

    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix" + suffix + "_1";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 1>);

    func_name = "get_boundary_matrix" + suffix + "_2";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 2>);

    func_name = "get_boundary_matrix" + suffix + "_3";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 3>);
}

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H