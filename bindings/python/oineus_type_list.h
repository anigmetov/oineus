#pragma once

// A minimal compile-time type list + fold, used to bind the per-cell-type filtration /
// decomposition / optimizer classes ONCE from a single function template instead of from
// near-identical hand macros (BIND_CUBE_FILTRATION / BIND_FR_FILTRATION / ...). The reduction
// core is cell-agnostic, but each Filtration<Cell> is a distinct C++ type that nanobind must
// bind separately; for_each_type instantiates the shared register function for every cell type
// in the list. The C++17 fold expression `(f.template operator()<Ts>(), ...)` is type-safe and
// needs no macros or recursion.

namespace oineus_python {

template<class... Ts>
struct TypeList {};

template<class... Ts, class F>
void for_each_type(TypeList<Ts...>, F&& f)
{
    (f.template operator()<Ts>(), ...);
}

} // namespace oineus_python
