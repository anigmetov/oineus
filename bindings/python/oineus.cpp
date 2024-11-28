#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "oineus_persistence_bindings.h"

PYBIND11_MODULE(_oineus, m)
{
    m.doc() = "Oineus python bindings";

    init_oineus_common(m);
    init_oineus_common_diagram(m);
    init_oineus_common_decomposition(m);

    init_oineus_functions(m);
    init_oineus_fil_dgm_simplex(m);

    init_oineus_top_optimizer(m);
}
