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

    PyOineusDiagrams(oineus::Diagrams<Real>&& _diagrams)
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
get_fr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto grid = get_grid<Int, Real, D>(data, wrap);
    auto fil =  grid.freudenthal_filtration(max_dim, negate, n_threads);
    return fil;
}

template<class Real, size_t D>
decltype(auto) numpy_to_point_vector(py::array_t<Real, py::array::c_style | py::array::forcecast> data)
{
    using PointVector = std::vector<oineus::Point<Real, D>>;

    if (data.ndim() != 2 or data.shape(1) != D)
        throw std::runtime_error("Dimension mismatch");

    py::buffer_info data_buf = data.request();

    PointVector points(data.shape(0));

    Real* pdata {static_cast<Real*>(data_buf.ptr)};

    for(size_t i = 0; i < data.size(); ++i)
        points[i / D][i % D] = pdata[i];

    return points;
}

template<class Int, class Real, size_t D>
typename oineus::Filtration<Int, Real, oineus::VREdge>
get_vr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> points, dim_type max_dim, Real max_radius, int n_threads)
{
    return oineus::get_vr_filtration_bk<Int, Real, D>(numpy_to_point_vector<Real, D>(points), max_dim, max_radius, n_threads);
}


template<class Int, class Real, size_t D>
typename oineus::SparseMatrix<Int>::MatrixData
get_boundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads);
    auto bm = fil.boundary_matrix_full();
    return bm.data;
}

template<class Int, class Real, size_t D>
DiagramV<Int, Real>
compute_diagrams_and_v_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim + 1, n_threads);
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
compute_diagrams_and_rv_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim + 1, n_threads);
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
compute_diagrams_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    // for diagram in dimension d, we need (d+1)-simplices
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim + 1, n_threads);
    auto d_matrix = fil.boundary_matrix_full();

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil));
}

template<class Int>
void init_oineus_common(py::module& m)
{
    using namespace pybind11::literals;

    using oineus::VREdge;

    using BoundaryMatrix = oineus::SparseMatrix<Int>;

    using ReductionParams = oineus::Params;

    std::string vr_edge_name = "VREdge";

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
            .def("diagram", [](const BoundaryMatrix& self, const oineus::Filtration<Int, double, VREdge>& fil) { return PyOineusDiagrams<double>(self.diagram(fil)); })
            .def("diagram", [](const BoundaryMatrix& self, const oineus::Filtration<Int, float, VREdge>& fil) { return PyOineusDiagrams<float>(self.diagram(fil)); })
            .def("diagram", [](const BoundaryMatrix& self, const oineus::Filtration<Int, double, Int>& fil) { return PyOineusDiagrams<double>(self.diagram(fil)); })
            .def("diagram", [](const BoundaryMatrix& self, const oineus::Filtration<Int, float, Int>& fil) { return PyOineusDiagrams<float>(self.diagram(fil)); })
            ;

    using DgmPointInt = typename oineus::DgmPoint<Int>;
    using DgmPointSizet = typename oineus::DgmPoint<size_t>;

    py::class_<DgmPointInt>(m, "DgmPoint_int")
            .def(py::init<Int, Int>())
            .def_readwrite("birth", &DgmPointInt::birth)
            .def_readwrite("death", &DgmPointInt::death)
            .def("__getitem__", [](const DgmPointInt& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPointInt& p) { return std::hash<DgmPointInt>()(p); })
            .def("__repr__", [](const DgmPointInt& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

     py::class_<DgmPointSizet>(m, "DgmPoint_Sizet")
            .def(py::init<size_t, size_t>())
            .def_readwrite("birth", &DgmPointSizet::birth)
            .def_readwrite("death", &DgmPointSizet::death)
            .def("__getitem__", [](const DgmPointSizet& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPointSizet& p) { return std::hash<DgmPointSizet>()(p); })
            .def("__repr__", [](const DgmPointSizet& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });
}

template<class Int, class Real>
void init_oineus(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using DgmPoint = oineus::DgmPoint<Real>;
    using Diagram = PyOineusDiagrams<Real>;

    using LSFiltration = oineus::Filtration<Int, Real, Int>;
    using LSSimplex = typename LSFiltration::FiltrationSimplex;

    using oineus::VREdge;
    using VRFiltration = oineus::Filtration<Int, Real, VREdge>;
    using VRSimplex = typename VRFiltration::FiltrationSimplex;

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
            .def("__getitem__", [](const DgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPoint& p) { return std::hash<DgmPoint>()(p); })
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<Diagram>(m, dgm_class_name.c_str())
            .def(py::init<dim_type>())
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

    py::class_<LSFiltration>(m, ls_filtration_class_name.c_str())
            .def(py::init<>())
            .def("max_dim", &LSFiltration::max_dim)
            .def("size_in_dimension", &LSFiltration::size_in_dimension)
            .def("boundary_matrix", &LSFiltration::boundary_matrix_full);

    py::class_<VRFiltration>(m, vr_filtration_class_name.c_str())
            .def(py::init<>())
            .def("max_dim", &VRFiltration::max_dim)
            .def("size_in_dimension", &VRFiltration::size_in_dimension)
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

    // Lower-star Freudenthal filtration
    func_name = "get_fr_filtration" + suffix + "_1";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 1>);

    func_name = "get_fr_filtration" + suffix + "_2";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 2>);

    func_name = "get_fr_filtration" + suffix + "_3";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 3>);

     // Vietoris--Rips filtration
    func_name = "get_vr_filtration" + suffix + "_1";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 1>);

    func_name = "get_vr_filtration" + suffix + "_2";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 2>);

    func_name = "get_vr_filtration" + suffix + "_3";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 3>);

    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix" + suffix + "_1";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 1>);

    func_name = "get_boundary_matrix" + suffix + "_2";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 2>);

    func_name = "get_boundary_matrix" + suffix + "_3";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 3>);

     // target values
    func_name = "get_denoise_target" + suffix;
    m.def(func_name.c_str(), &oineus::get_denoise_target<Int, Real, Int>);

    func_name = "get_denoise_target" + suffix;
    m.def(func_name.c_str(), &oineus::get_denoise_target<Int, Real, VREdge>);

    // target values
    func_name = "get_ls_target_values" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_values<Int, Real, Int>);

    func_name = "get_vr_target_values" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_values<Int, Real, VREdge>);

     // target values -- diagram loss
    func_name = "get_ls_target_values_diagram_loss" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_values_diagram_loss<Int, Real, Int>);

    func_name = "get_vr_target_values_diagram_loss" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_values_diagram_loss<Int, Real, VREdge>);

}

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H