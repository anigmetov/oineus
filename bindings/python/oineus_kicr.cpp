#include "oineus_persistence_bindings.h"

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

    nb::class_<KerImCokRedSimplex>(m, ker_im_cok_reduced_class_name.c_str())
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
            .def_rw("fil_K", &KerImCokRedSimplex::fil_K_)
            .def_rw("fil_L", &KerImCokRedSimplex::fil_L_)
            ;

     nb::class_<KerImCokRedProdSimplex>(m, ker_im_cok_reduced_prod_class_name.c_str())
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
}