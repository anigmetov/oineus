#include "oineus_persistence_bindings.h"

void init_oineus_kicr(py::module& m)
{
    using namespace pybind11::literals;

    using DgmPoint = typename oin::Diagrams<oin_real>::Point;
    using DgmPtVec = typename oin::Diagrams<oin_real>::Dgm;
    using IndexDgmPtVec = typename oin::Diagrams<oin_real>::IndexDgm;
    using Diagram = PyOineusDiagrams<oin_real>;

    using Simplex = oin::Simplex<oin_int>;
    using SimplexValue = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>;
    using Filtration = oin::Filtration<Simplex, oin_real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexValue = oin::CellWithValue<ProdSimplex, oin_real>;
    using ProdFiltration = oin::Filtration<ProdSimplex, oin_real>;

    using KerImCokRedSimplex = oin::KerImCokReduced<Simplex, oin_real>;
    using KerImCokRedProdSimplex = oin::KerImCokReduced<ProdSimplex, oin_real>;

    const std::string ker_im_cok_reduced_class_name = "KerImCokReduced";
    const std::string ker_im_cok_reduced_prod_class_name = "KerImCokReducedProd";

    py::class_<KerImCokRedSimplex>(m, ker_im_cok_reduced_class_name.c_str())
            .def(py::init<const Filtration&, const Filtration&, oin::KICRParams&>(), py::arg("K"), py::arg("L"), py::arg("params"))
            .def("domain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_domain_diagrams()); })
            .def("codomain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_codomain_diagrams()); })
            .def("kernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_readwrite("decomposition_f", &KerImCokRedSimplex::dcmp_F_)
            .def_readwrite("decomposition_g", &KerImCokRedSimplex::dcmp_G_)
            .def_readwrite("decomposition_im", &KerImCokRedSimplex::dcmp_im_)
            .def_readwrite("decomposition_ker", &KerImCokRedSimplex::dcmp_ker_)
            .def_readwrite("decomposition_cok", &KerImCokRedSimplex::dcmp_cok_)
            ;

     py::class_<KerImCokRedProdSimplex>(m, ker_im_cok_reduced_prod_class_name.c_str())
            .def(py::init<const ProdFiltration&, const ProdFiltration&, oin::KICRParams&>(), py::arg("K"), py::arg("L"), py::arg("params"))
            .def("kernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<oin_real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_readwrite("decomposition_f", &KerImCokRedProdSimplex::dcmp_F_)
            .def_readwrite("decomposition_g", &KerImCokRedProdSimplex::dcmp_G_)
            .def_readwrite("decomposition_im", &KerImCokRedProdSimplex::dcmp_im_)
            .def_readwrite("decomposition_ker", &KerImCokRedProdSimplex::dcmp_ker_)
            .def_readwrite("decomposition_cok", &KerImCokRedProdSimplex::dcmp_cok_)
            ;
}