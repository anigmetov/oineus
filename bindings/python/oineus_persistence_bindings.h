#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <oineus/includes.h>

using dim_type = oineus::dim_type;

template<class Int, class Real, size_t D>
oineus::Diagram<Real> compute_diagrams(py::array_t<Real> data, bool negate, bool wrap, dim_type top_d, oineus::Params params)
{
    using Grid = oineus::Grid<Int, Real, D>;
    using GridPoint = typename Grid::GridPoint;

    if (data.ndim()!=D)
        throw std::runtime_error("Dimension mismatch");

    GridPoint dims;

    for (dim_type d = 0; d<D; ++d)
        dims[d] = data.shape(d);

    Real* pdata{static_cast<Real*>(data.ptr())};
    bool c_order{true};
    Grid grid{dims, wrap, pdata};
    auto fil = grid.freudenthal_filtration(top_d, negate);
    auto bm = fil.boundary_matrix_full();
    bm.reduce_parallel(params);
    return bm.diagram(fil);
}

template<class Int, class Real>
void init_oineus(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using RealDgmPoint = oineus::DgmPoint<Real>;
    using RealDiagram = oineus::Diagram<Real>;

    std::string dgm_point_name = "DiagramPoint"+suffix;
    std::string dgm_class_name = "Diagram"+suffix;

    py::class_<RealDgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<Real, Real>())
            .def_readwrite("birth", &RealDgmPoint::birth)
            .def_readwrite("death", &RealDgmPoint::death)
            .def("__repr__", [](const RealDgmPoint& p) {
                std::stringstream ss;
                ss << p;
                return ss.str();
            });

    py::class_<RealDiagram>(m, dgm_class_name.c_str())
            .def(py::init<>())
            .def("in_dimension", &RealDiagram::get_diagram_in_dimension)
            .def("add_point", &RealDiagram::add_point)
            .def("sort", &RealDiagram::sort)
            .def("save_as_txt", &RealDiagram::save_as_txt);
}

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
