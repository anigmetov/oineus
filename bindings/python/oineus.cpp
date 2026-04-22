#include "oineus_persistence_bindings.h"

NB_MODULE(_oineus, m)
{
    m.doc() = "Oineus python bindings";

    if constexpr (std::is_same_v<oin_real, float>)
        m.attr("real_dtype") = "float32";
    else
        m.attr("real_dtype") = "float64";

    init_oineus_common(m);
    init_oineus_cells(m);
    init_oineus_filtration(m);
    init_oineus_common_decomposition(m);
    init_oineus_diagram(m);
    init_oineus_functions(m);
    init_oineus_kicr(m);
    init_oineus_top_optimizer(m);
}
