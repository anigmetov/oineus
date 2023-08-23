#include "oineus_persistence_bindings.h"

void init_oineus_functions_double(py::module& m, std::string suffix)
{
    init_oineus_functions<int, double>(m, suffix);
}
