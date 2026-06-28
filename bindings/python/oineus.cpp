#include "oineus_persistence_bindings.h"

NB_MODULE(_oineus, m)
{
    m.doc() = "Oineus python bindings";

    oineus_python::register_interrupt_translator();

    // Real-INDEPENDENT registration: enums, parameter structs, the int-templated
    // decomposition stats. Registered once on the top module.
    init_oineus_common(m);
    init_oineus_dcmp_stats(m);

    // float64 (the default Real): registered on the TOP module, byte-identical to the
    // historical single-dtype layout. reg_indep=true also registers the few
    // Real-INDEPENDENT classes that live alongside the Real-dependent ones
    // (CombinatorialSimplex / ProdSimplex, GridDomain, CombinatorialCube, the
    // Decomposition class, UStrategy, IndexDiagramPoint).
    register_oineus_cells<double>(m, /*reg_indep=*/true);
    register_oineus_filtration<double>(m, true);
    register_oineus_decomposition<double>(m, true);
    register_oineus_diagram<double>(m, true);
    register_oineus_kicr<double>(m, true);
    register_oineus_top_optimizer<double>(m, true);
    // Hera distances / matchings are numerically dtype-agnostic, so this stays
    // double-only; the Python facade upcasts float32 diagrams for distance calls.
    init_oineus_functions(m);

    // float32: the Real-dependent types are registered into the _f32 submodule with
    // reg_indep=false -- the shared Real-independent types above are found in
    // nanobind's process-global type registry. Class NAMES are identical to the
    // top-module float64 ones; the dtype lives in the submodule path, hidden by the
    // Python facade. The Decomposition class is shared: register_oineus_decomposition
    // adds the float ctor/diagram overloads to the single top-module class.
    nb::module_ f32 = m.def_submodule("_f32", "float32 variants of the Real-dependent types");
    register_oineus_cells<float>(f32, /*reg_indep=*/false);
    register_oineus_filtration<float>(f32, false);
    register_oineus_decomposition<float>(f32, false);
    register_oineus_diagram<float>(f32, false);
    register_oineus_kicr<float>(f32, false);
    register_oineus_top_optimizer<float>(f32, false);

    m.attr("real_dtype") = "float64";          // back-compat: the default Real
    f32.attr("real_dtype") = "float32";
    m.attr("real_dtypes") = nb::make_tuple("float64", "float32");

    // both passes are done; drop the shared Decomposition class handle held during init
    finalize_oineus_decomposition();
}
