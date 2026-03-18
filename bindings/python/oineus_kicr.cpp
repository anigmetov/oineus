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

void init_oineus_kicr(nb::module_& m)
{
    using Simplex = oin::Simplex<oin_int>;
    using Filtration = oin::Filtration<Simplex, oin_real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdFiltration = oin::Filtration<ProdSimplex, oin_real>;

    using KerImCokRedSimplex = oin::KerImCokReduced<Simplex, oin_real>;
    using KerImCokRedProdSimplex = oin::KerImCokReduced<ProdSimplex, oin_real>;

    const std::string ker_im_cok_reduced_class_name = "KerImCokReduced";
    const std::string ker_im_cok_reduced_prod_class_name = "KerImCokReducedProd";

    auto kicr_simplex = nb::class_<KerImCokRedSimplex>(m, ker_im_cok_reduced_class_name.c_str())
            .def(nb::init<const Filtration&, const Filtration&, oin::KICRParams&>(), nb::arg("K"), nb::arg("L"), nb::arg("params"))
            .def("domain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_domain_diagrams()); })
            .def("codomain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_codomain_diagrams()); })
            .def("kernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
			.def("old_order_to_new", [](const KerImCokRedSimplex& self) { return self.get_old_order_to_new(); })
			.def("new_order_to_old", [](const KerImCokRedSimplex& self) { return self.get_new_order_to_old(); })
			.def_rw("fil_K", &KerImCokRedSimplex::fil_K_)
			.def_rw("fil_L", &KerImCokRedSimplex::fil_L_)
            // decomposition objects provide access to their R/V/U matrices
            .def_rw("decomposition_f", &KerImCokRedSimplex::dcmp_F_)
            .def_rw("decomposition_g", &KerImCokRedSimplex::dcmp_G_)
            .def_rw("decomposition_im", &KerImCokRedSimplex::dcmp_im_)
            .def_rw("decomposition_ker", &KerImCokRedSimplex::dcmp_ker_)
            .def_rw("decomposition_cok", &KerImCokRedSimplex::dcmp_cok_)
            ;
    bind_kicr_pickle_and_equality(kicr_simplex);

    auto kicr_prod = nb::class_<KerImCokRedProdSimplex>(m, ker_im_cok_reduced_prod_class_name.c_str())
            .def(nb::init<const ProdFiltration&, const ProdFiltration&, oin::KICRParams&>(), nb::arg("K"), nb::arg("L"), nb::arg("params"))
            .def("kernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_rw("decomposition_f", &KerImCokRedProdSimplex::dcmp_F_)
            .def_rw("decomposition_g", &KerImCokRedProdSimplex::dcmp_G_)
            .def_rw("decomposition_im", &KerImCokRedProdSimplex::dcmp_im_)
            .def_rw("decomposition_ker", &KerImCokRedProdSimplex::dcmp_ker_)
            .def_rw("decomposition_cok", &KerImCokRedProdSimplex::dcmp_cok_)
            .def_rw("fil_K", &KerImCokRedProdSimplex::fil_K_)
            .def_rw("fil_L", &KerImCokRedProdSimplex::fil_L_)
            ;
    bind_kicr_pickle_and_equality(kicr_prod);
}
