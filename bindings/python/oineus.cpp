#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "oineus_persistence_bindings.h"

PYBIND11_MODULE(_oineus, m)
{
    m.doc() = "Oineus python bindings";

//    std::string float_suffix = "_float";
    std::string double_suffix = "_double";
//    std::string double_suffix = "";

    init_oineus_common_int(m);
    init_oineus_common_diagram_int(m);
    init_oineus_common_decomposition_int(m);

//    init_oineus_functions_float(m, float_suffix);
//    init_oineus_fil_dgm_simplex_float(m, float_suffix);

    init_oineus_functions_double(m, double_suffix);
    init_oineus_fil_dgm_simplex_double(m, double_suffix);

    init_oineus_top_optimizer(m);
}
