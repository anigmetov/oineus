#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#include <iostream>
#include <vector>
#include <sstream>
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

template<class Int, class Real>
using DiagramRV = std::tuple<PyOineusDiagrams<Real>, typename oineus::SparseMatrix<Int>::MatrixData, typename oineus::SparseMatrix<Int>::MatrixData>;

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
typename oineus::Filtration<Int, Real, Int>
get_fr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto grid = get_grid<Int, Real, D>(data, wrap);
    return grid.freudenthal_filtration(top_d, negate, n_threads);
}

template<class Int, class Real, size_t D>
typename oineus::SparseMatrix<Int>::MatrixData
get_boundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
    auto bm = fil.boundary_matrix_full();
    return bm.data;
}

template<class Int, class Real, size_t D>
DiagramV<Int, Real>
compute_diagrams_and_v_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
    auto d_matrix = fil.boundary_matrix_full();

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = false;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return {PyOineusDiagrams<Real>(d_matrix.diagram(fil)), d_matrix.v_data};
}

template<class Int, class Real, size_t D>
DiagramRV<Int, Real>
compute_diagrams_and_rv_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
    auto d_matrix = fil.boundary_matrix_full();

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return {PyOineusDiagrams<Real>(d_matrix.diagram(fil)), d_matrix.data, d_matrix.v_data};
}

template<class Int, class Real, size_t D>
PyOineusDiagrams<Real>
compute_diagrams_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
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

    using DgmPoint = oineus::DgmPoint<Real>;
    using Diagram = PyOineusDiagrams<Real>;

    using LSFiltration = oineus::Filtration<Int, Real, Int>;
    using LSSimplex = typename LSFiltration::FiltrationSimplex;

    using VREdge = typename oineus::VREdge<Int>;
    using VRFiltration = oineus::Filtration<Int, Real, VREdge>;
    using VRSimplex = typename VRFiltration::FiltrationSimplex;

    using BoundaryMatrix = oineus::SparseMatrix<Int>;

    using ReductionParams = oineus::Params;

    std::string vr_edge_name = "VREdge";

    std::string dgm_point_name = "DiagramPoint" + suffix;
    std::string dgm_class_name = "Diagrams" + suffix;

    std::string ls_simplex_class_name = "LSSimplex" + suffix;
    std::string ls_filtration_class_name = "LSFiltration" + suffix;

    std::string vr_simplex_class_name = "VRSimplex" + suffix;
    std::string vr_filtration_class_name = "VRFiltration" + suffix;

    py::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<Real, Real>())
            .def_readwrite("birth", &DgmPoint::birth)
            .def_readwrite("death", &DgmPoint::death)
            .def("__getitem__", [](const DgmPoint& p, int i) {
              if (i == 0)
                  return p.birth;
              else if (i == 1)
                  return p.death;
              else
                  throw std::out_of_range("i must be 0 or 1");
            })
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<VREdge>(m, vr_edge_name.c_str())
            .def(py::init<Int>())
            .def_readwrite("x", &VREdge::x)
            .def_readwrite("y", &VREdge::y)
            .def("__getitem__", [](const VREdge& p, int i) {
              if (i == 0)
                  return p.x;
              else if (i == 1)
                  return p.y;
              else
                  throw std::out_of_range("i must be 0 or 1");
            })
            .def("__repr__", [](const VREdge& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });


    py::class_<Diagram>(m, dgm_class_name.c_str())
            .def(py::init<>())
            .def("in_dimension", &Diagram::get_diagram_in_dimension)
            .def("__getitem__", &Diagram::get_diagram_in_dimension);

    py::class_<LSSimplex>(m, ls_simplex_class_name.c_str())
            .def(py::init<>())
            .def_readwrite("id", &LSSimplex::id_)
            .def_readwrite("sorted_id", &LSSimplex::sorted_id_)
            .def_readwrite("vertices", &LSSimplex::vertices_)
            .def_readwrite("value", &LSSimplex::value_)
            .def_readwrite("critical_vertex", &LSSimplex::critical_value_location_)
            .def("dim", &LSSimplex::dim)
            .def("__repr__", [](const LSSimplex& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

     py::class_<VRSimplex>(m, vr_simplex_class_name.c_str())
            .def(py::init<>())
            .def_readwrite("id", &VRSimplex::id_)
            .def_readwrite("sorted_id", &VRSimplex::sorted_id_)
            .def_readwrite("vertices", &VRSimplex::vertices_)
            .def_readwrite("value", &VRSimplex::value_)
            .def_readwrite("critical_edge", &VRSimplex::critical_value_location_)
            .def("dim", &VRSimplex::dim)
            .def("__repr__", [](const VRSimplex& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    py::class_<ReductionParams>(m, "ReductionParams")
            .def(py::init<>())
            .def_readwrite("n_threads", &ReductionParams::n_threads)
            .def_readwrite("chunk_size", &ReductionParams::chunk_size)
            .def_readwrite("write_dgms", &ReductionParams::write_dgms)
            .def_readwrite("sort_dgms", &ReductionParams::sort_dgms)
            .def_readwrite("clearing_opt", &ReductionParams::clearing_opt)
            .def_readwrite("acq_rel", &ReductionParams::acq_rel)
            .def_readwrite("print_time", &ReductionParams::print_time)
            .def_readwrite("elapsed", &ReductionParams::elapsed)
            ;

    py::class_<BoundaryMatrix>(m, "BoundaryMatrix")
            .def(py::init<>())
            .def_readwrite("data", &BoundaryMatrix::data)
            .def_readwrite("v_data", &BoundaryMatrix::v_data)
            .def("reduce", &BoundaryMatrix::reduce_parallel)
            ;

    py::class_<LSFiltration>(m, ls_filtration_class_name.c_str())
            .def(py::init<>())
            .def("simplices", &LSFiltration::simplices)
            .def("boundary_matrix", &LSFiltration::boundary_matrix_full);

    py::class_<VRFiltration>(m, vr_filtration_class_name.c_str())
            .def(py::init<>())
            .def("simplices", &VRFiltration::simplices)
            .def("boundary_matrix", &VRFiltration::boundary_matrix_full);

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

    // diagrams and R,V matrices
    func_name = "compute_diagrams_and_rv_ls" + suffix + "_1";
    m.def(func_name.c_str(), &compute_diagrams_and_rv_ls_freudenthal<Int, Real, 1>);

    func_name = "compute_diagrams_and_rv_ls" + suffix + "_2";
    m.def(func_name.c_str(), &compute_diagrams_and_rv_ls_freudenthal<Int, Real, 2>);

    func_name = "compute_diagrams_and_rv_ls" + suffix + "_3";
    m.def(func_name.c_str(), &compute_diagrams_and_rv_ls_freudenthal<Int, Real, 3>);

    // boundary matrix as vector of columns
    func_name = "get_fr_filtration" + suffix + "_1";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 1>);

    func_name = "get_fr_filtration" + suffix + "_2";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 2>);

    func_name = "get_fr_filtration" + suffix + "_3";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 3>);

    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix" + suffix + "_1";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 1>);

    func_name = "get_boundary_matrix" + suffix + "_2";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 2>);

    func_name = "get_boundary_matrix" + suffix + "_3";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 3>);

    // target values
    func_name = "get_ls_target_values" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_values<Int, Real, Int>);

    func_name = "get_vr_target_values" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_values<Int, Real, VREdge>);
}

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H