#include "oineus_persistence_bindings.h"

template<class KerImCokReduced>
void bind_kicr_pickle_and_equality(nb::class_<KerImCokReduced>& cls)
{
    using DiagramState = decltype(KerImCokReduced::dom_diagrams_.diagram_in_dimension_);
    using KICRStateTuple = std::tuple<decltype(KerImCokReduced::fil_K_),
                                      decltype(KerImCokReduced::fil_L_),
                                      decltype(KerImCokReduced::dcmp_F_),
                                      decltype(KerImCokReduced::dcmp_F_D_),
                                      decltype(KerImCokReduced::dcmp_G_),
                                      decltype(KerImCokReduced::dcmp_im_),
                                      decltype(KerImCokReduced::dcmp_ker_),
                                      decltype(KerImCokReduced::dcmp_cok_),
                                      decltype(KerImCokReduced::max_dim_),
                                      DiagramState,
                                      DiagramState,
                                      DiagramState,
                                      DiagramState,
                                      DiagramState,
                                      decltype(KerImCokReduced::sorted_K_to_sorted_L_),
                                      decltype(KerImCokReduced::sorted_L_to_sorted_K_),
                                      decltype(KerImCokReduced::new_order_to_old_),
                                      decltype(KerImCokReduced::old_order_to_new_),
                                      decltype(KerImCokReduced::K_to_ker_column_index_),
                                      decltype(KerImCokReduced::params_)>;

    cls.def(nb::self == nb::self)
       .def(nb::self != nb::self)
       .def("__getstate__", [](const KerImCokReduced& self) -> KICRStateTuple {
            return std::make_tuple(self.fil_K_, self.fil_L_, self.dcmp_F_, self.dcmp_F_D_, self.dcmp_G_,
                    self.dcmp_im_, self.dcmp_ker_, self.dcmp_cok_, self.max_dim_,
                    self.dom_diagrams_.diagram_in_dimension_, self.cod_diagrams_.diagram_in_dimension_,
                    self.ker_diagrams_.diagram_in_dimension_, self.im_diagrams_.diagram_in_dimension_,
                    self.cok_diagrams_.diagram_in_dimension_,
                    self.sorted_K_to_sorted_L_, self.sorted_L_to_sorted_K_, self.new_order_to_old_,
                    self.old_order_to_new_, self.K_to_ker_column_index_, self.params_);
        })
       .def("__setstate__", [](KerImCokReduced& self, const KICRStateTuple& t) {
            auto params = std::get<19>(t);
            new (&self) KerImCokReduced(std::get<0>(t), std::get<1>(t), params);
            self.fil_K_ = std::get<0>(t);
            self.fil_L_ = std::get<1>(t);
            self.dcmp_F_ = std::get<2>(t);
            self.dcmp_F_D_ = std::get<3>(t);
            self.dcmp_G_ = std::get<4>(t);
            self.dcmp_im_ = std::get<5>(t);
            self.dcmp_ker_ = std::get<6>(t);
            self.dcmp_cok_ = std::get<7>(t);
            self.max_dim_ = std::get<8>(t);
            self.dom_diagrams_.diagram_in_dimension_ = std::get<9>(t);
            self.cod_diagrams_.diagram_in_dimension_ = std::get<10>(t);
            self.ker_diagrams_.diagram_in_dimension_ = std::get<11>(t);
            self.im_diagrams_.diagram_in_dimension_ = std::get<12>(t);
            self.cok_diagrams_.diagram_in_dimension_ = std::get<13>(t);
            self.sorted_K_to_sorted_L_ = std::get<14>(t);
            self.sorted_L_to_sorted_K_ = std::get<15>(t);
            self.new_order_to_old_ = std::get<16>(t);
            self.old_order_to_new_ = std::get<17>(t);
            self.K_to_ker_column_index_ = std::get<18>(t);
            self.params_ = std::get<19>(t);
        });
}

// Bind KerImCokReduced<Cell> under a given Python name. kernel.h is cell-agnostic (it builds
// boundary matrices via fil.boundary_matrix() and works purely on uids/dims/sorted_ids), so the
// same class body works for every cell type -- fat Simplex / product, slim Freudenthal, bit-packed
// VR/alpha, slim cube. Mirrors init_oineus_top_optimizer_class<Cell> in oineus_top_optimizer.cpp.
template<class Cell, class Real>
void init_oineus_kicr_class(nb::module_& m, const std::string& class_name)
{
    using oin_real = Real;
    using Fil = oin::Filtration<Cell, oin_real>;
    using KICR = oin::KerImCokReduced<Cell, oin_real>;

    auto cls = nb::class_<KICR>(m, class_name.c_str())
            .def(nb::init<const Fil&, const Fil&, oin::KICRParams&>(), nb::arg("K"), nb::arg("L"), nb::arg("params"),
                 nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("domain_diagrams", [](const KICR& self) { return PyOineusDiagrams<oin_real>(self.get_domain_diagrams()); })
            .def("codomain_diagrams", [](const KICR& self) { return PyOineusDiagrams<oin_real>(self.get_codomain_diagrams()); })
            .def("kernel_diagrams", [](const KICR& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KICR& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KICR& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
            .def("old_order_to_new", [](const KICR& self) { return self.get_old_order_to_new(); })
            .def("new_order_to_old", [](const KICR& self) { return self.get_new_order_to_old(); })
            .def_rw("fil_K", &KICR::fil_K_)
            .def_rw("fil_L", &KICR::fil_L_)
            // decomposition objects provide access to their R/V/U matrices
            .def_rw("decomposition_f", &KICR::dcmp_F_)
            .def_rw("decomposition_g", &KICR::dcmp_G_)
            .def_rw("decomposition_im", &KICR::dcmp_im_)
            .def_rw("decomposition_ker", &KICR::dcmp_ker_)
            .def_rw("decomposition_cok", &KICR::dcmp_cok_)
            .def("__str__", [](const KICR& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def("__repr__", [](const KICR& self) { std::stringstream ss; ss << self; return ss.str(); })
            ;
    bind_kicr_pickle_and_equality(cls);
}

template<class Real>
void register_oineus_kicr(nb::module_& m, bool reg_indep)
{
    using oin_real = Real;
    using Simp = oin::Simplex<oin_int>;
    using SimpProd = oin::ProductCell<Simp, Simp>;
    using Cube_1D = oin::Cube<oin_int, 1>;
    using Cube_2D = oin::Cube<oin_int, 2>;
    using Cube_3D = oin::Cube<oin_int, 3>;
    using Cube_4D = oin::Cube<oin_int, 4>;
    using FrCell_1D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 1>>;
    using FrCell_2D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 2>>;
    using FrCell_3D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 3>>;
    using FrCell_4D = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 4>>;
    using PackedCell_64 = oin::Simplex<oin_int, oin::BitPacked<oin_int, std::uint64_t>>;
    using PackedCell_128 = oin::Simplex<oin_int, oin::BitPacked<oin_int, unsigned __int128>>;

    // The two historical fat classes keep their public names (imported by name in __init__.py
    // and used directly in compute_ker_cok_reduction_cyl); the slim/packed/cube families get
    // hidden underscore names dispatched via _KICR_CLASS_BY_FIL_TYPE in __init__.py.
    init_oineus_kicr_class<Simp, oin_real>(m, "KerImCokReduced");
    init_oineus_kicr_class<SimpProd, oin_real>(m, "KerImCokReducedProd");
    init_oineus_kicr_class<Cube_1D, oin_real>(m, "_KerImCokReduced_Cube_1D");
    init_oineus_kicr_class<Cube_2D, oin_real>(m, "_KerImCokReduced_Cube_2D");
    init_oineus_kicr_class<Cube_3D, oin_real>(m, "_KerImCokReduced_Cube_3D");
    init_oineus_kicr_class<Cube_4D, oin_real>(m, "_KerImCokReduced_Cube_4D");
    init_oineus_kicr_class<FrCell_1D, oin_real>(m, "_KerImCokReduced_Fr_1D");
    init_oineus_kicr_class<FrCell_2D, oin_real>(m, "_KerImCokReduced_Fr_2D");
    init_oineus_kicr_class<FrCell_3D, oin_real>(m, "_KerImCokReduced_Fr_3D");
    init_oineus_kicr_class<FrCell_4D, oin_real>(m, "_KerImCokReduced_Fr_4D");
    init_oineus_kicr_class<PackedCell_64, oin_real>(m, "_KerImCokReduced_Packed_64");
    init_oineus_kicr_class<PackedCell_128, oin_real>(m, "_KerImCokReduced_Packed_128");
}

// double pass on the top module (all KICR types are Real-dependent, no reg_indep)
void init_oineus_kicr(nb::module_& m) { register_oineus_kicr<double>(m, true); }

// float pass is compiled here; the driver calls it into the _f32 submodule
template void register_oineus_kicr<float>(nb::module_&, bool);
