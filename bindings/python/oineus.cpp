#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "oineus_persistence_bindings.h"
#include <oineus/oineus.h>

PYBIND11_MODULE(_oineus, m)
{
    m.doc() = "Oineus python bindings";

    init_oineus<int, float>(m, "_float");
    init_oineus<int, double>(m, "_double");

//    init_vectorizer<float>(m, "_float");
//    init_vectorizer<double>(m, "_double");
}
