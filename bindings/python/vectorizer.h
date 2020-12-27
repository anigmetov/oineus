#pragma once

#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


template<class Real>
std::vector<Real> persfunc_unstable(py::array_t<Real> points, py::array_t<Real> diagram, Real sigma)
{
    bool verbose = false;

    auto eval_points = points.template unchecked<2>();
    auto dgm_points = diagram.template unchecked<2>();

    if (eval_points.shape(1) != 2) {
        throw std::runtime_error("points must be an Nx2 NumPy array");
    }

    if (dgm_points.shape(1) != 2) {
        throw std::runtime_error("diagram must be an Nx2 NumPy array");
    }

    size_t n_eval_points = eval_points.shape(0);
    size_t n_dgm_points = dgm_points.shape(0);

    std::vector<Real> result(n_eval_points, Real(0));

    auto start = std::chrono::steady_clock::now();
    for(size_t eval_point_idx = 0; eval_point_idx < n_eval_points; ++eval_point_idx) {

        for(size_t dgm_point_idx = 0; dgm_point_idx < n_dgm_points; ++dgm_point_idx) {

            Real xdiff = eval_points(eval_point_idx, 0) - dgm_points(dgm_point_idx, 0);
            Real ydiff = eval_points(eval_point_idx, 1) - dgm_points(dgm_point_idx, 1);
            Real sqdist = xdiff * xdiff + ydiff * ydiff;

            result[eval_point_idx] += exp(-sqdist / sigma);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (verbose)
        std::cerr << "diagram points: " << n_dgm_points << ", pers. image time = " << elapsed.count() << std::endl;

    return result;
}

template<class Real>
std::vector<Real> persfunc_stable(py::array_t<Real> points, py::array_t<Real> diagram, Real sigma)
{
    auto eval_points = points.template unchecked<2>();
    auto dgm_points = diagram.template unchecked<2>();

    if (eval_points.shape(1) != 2) {
        throw std::runtime_error("points must be an Nx2 NumPy array");
    }

    if (dgm_points.shape(1) != 2) {
        throw std::runtime_error("diagram must be an Nx2 NumPy array");
    }

    size_t n_eval_points = eval_points.shape(0);
    size_t n_dgm_points = dgm_points.shape(0);

    std::vector<Real> result(n_eval_points, Real(0));

    for(size_t eval_point_idx = 0; eval_point_idx < n_eval_points; ++eval_point_idx) {

        for(size_t dgm_point_idx = 0; dgm_point_idx < n_dgm_points; ++dgm_point_idx) {
            Real xdiff = eval_points(eval_point_idx, 0) - dgm_points(dgm_point_idx, 0);
            Real ydiff = eval_points(eval_point_idx, 1) - dgm_points(dgm_point_idx, 1);
            Real sqdist = xdiff * xdiff + ydiff * ydiff;

            Real persistence = fabs(dgm_points(dgm_point_idx, 1) - dgm_points(dgm_point_idx, 0));

            result[eval_point_idx] +=  persistence * exp(-sqdist / sigma);
        }
    }

    return result;
}


template<class Real>
void init_vectorizer(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    std::string func_name = "persfunc_unstable" + suffix;

    m.def(func_name.c_str(),
          [](py::array_t<Real> points, py::array_t<Real> diagram, Real sigma) {
                 return persfunc_unstable<Real>(points, diagram, sigma);
            },
          py::call_guard<py::gil_scoped_release>());

    func_name = "persfunc_stable" + suffix;
    m.def(func_name.c_str(),
          [](py::array_t<Real> points, py::array_t<Real> diagram, Real sigma) {
              return persfunc_stable<Real>(points, diagram, sigma);
             },
          py::call_guard<py::gil_scoped_release>());
}
