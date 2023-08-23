#include "oineus_persistence_bindings.h"

void init_oineus_common_diagram_int(py::module& m)
{
    init_oineus_common_diagram<int>(m);
}
