#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "vectorizer.h"

void init_vectorizer(py::module&, std::string);

PYBIND11_MODULE(_smilis, m)
{
    m.doc() = "Smilis python bindings";

    init_vectorizer<double>(m, "_float");
    init_vectorizer<double>(m, "_double");
}

