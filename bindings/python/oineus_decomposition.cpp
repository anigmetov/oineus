#include "oineus_persistence_bindings.h"

void init_oineus_common_decomposition_int(py::module& m)
{
    init_oineus_common_decomposition<int>(m);
}
