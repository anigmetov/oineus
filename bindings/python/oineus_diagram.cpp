#include "oineus_persistence_bindings.h"
#include <nanobind/stl/variant.h>

void init_oineus_diagram(nb::module_& m)
{
    using DgmPoint = typename oin::Diagrams<oin_real>::Point;
    using DgmPtVec = typename oin::Diagrams<oin_real>::Dgm;
    using IndexDgmPoint = typename oin::Diagrams<size_t>::Point;
    using IndexDgmPtVec = typename oin::Diagrams<oin_real>::IndexDgm;
    using Diagram = PyOineusDiagrams<oin_real>;

    const std::string dgm_point_name = "DiagramPoint";
    const std::string index_dgm_point_name = "IndexDiagramPoint";
    const std::string dgm_class_name = "Diagrams";

    nb::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(nb::init<oin_real, oin_real>(), nb::arg("birth"), nb::arg("death"))
            .def_rw("birth", &DgmPoint::birth)
            .def_rw("death", &DgmPoint::death)
            .def_rw("birth_index", &DgmPoint::birth_index)
            .def_rw("death_index", &DgmPoint::death_index)
            .def_prop_ro("persistence", &DgmPoint::persistence)
            .def_prop_ro("index_persistence", &DgmPoint::persistence)
            .def("is_diagonal", &DgmPoint::is_diagonal)
            .def("is_inf", &DgmPoint::is_inf)
            .def("__getitem__", [](const DgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPoint& p) { return std::hash<DgmPoint>()(p); })
            .def("__eq__", [](const DgmPoint& p, const DgmPoint& q) { return p == q; })
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    nb::class_<IndexDgmPoint>(m, index_dgm_point_name.c_str())
            .def(nb::init<size_t, size_t>())
            .def_rw("birth", &IndexDgmPoint::birth)
            .def_rw("death", &IndexDgmPoint::death)
            .def_prop_ro("persistence", [](const IndexDgmPoint& p)
                 { return std::abs(static_cast<long long>(p.birth) - static_cast<long long>(p.death)); })
            .def("__getitem__", [](const IndexDgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const IndexDgmPoint& p) { return std::hash<IndexDgmPoint>()(p); })
            .def("__eq__", [](const IndexDgmPoint& p, const IndexDgmPoint& q) { return p == q; })
            .def("__repr__", [](const IndexDgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    nb::class_<Diagram>(m, dgm_class_name.c_str())
            .def(nb::init<dim_type>())
            .def("in_dimension", [](Diagram& self, dim_type dim, bool as_numpy) -> std::variant<nb::ndarray<oin_real>, DgmPtVec> {
                      if (as_numpy)
                          return self.get_diagram_in_dimension_as_numpy(dim);
                      else
                          return self.get_diagram_in_dimension(dim);
                    }, "return persistence diagram in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    nb::arg("dim"), nb::arg("as_numpy") = true)
            .def("index_diagram_in_dimension", [](Diagram& self, dim_type dim, bool as_numpy, bool sorted) -> std::variant<nb::ndarray<size_t>, IndexDgmPtVec> {
                      if (as_numpy)
                          return self.get_index_diagram_in_dimension_as_numpy(dim, sorted);
                      else
                          return self.get_index_diagram_in_dimension(dim, sorted);
                    }, "return persistence pairing (index diagram) in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    nb::arg("dim"), nb::arg("as_numpy") = true, nb::arg("sorted") =  true)
            .def("__getitem__", &Diagram::get_diagram_in_dimension_as_numpy);
}
