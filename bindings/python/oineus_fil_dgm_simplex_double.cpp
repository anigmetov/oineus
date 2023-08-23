#include "oineus_persistence_bindings.h"

void init_oineus_fil_dgm_simplex_double(py::module& m, std::string suffix)
{
    init_oineus_fil_dgm_simplex<int, double>(m, suffix);
}
