#include "oineus_persistence_bindings.h"

void init_oineus_diagram(py::module& m)
{
    using namespace pybind11::literals;

    using DgmPoint = typename oin::Diagrams<oin_real>::Point;
    using DgmPtVec = typename oin::Diagrams<oin_real>::Dgm;
    using IndexDgmPoint = typename oin::Diagrams<size_t>::Point;
    using IndexDgmPtVec = typename oin::Diagrams<oin_real>::IndexDgm;
    using Diagram = PyOineusDiagrams<oin_real>;

    const std::string dgm_point_name = "DiagramPoint";
    const std::string index_dgm_point_name = "IndexDiagramPoint";
    const std::string dgm_class_name = "Diagrams";

    py::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<oin_real, oin_real>(), py::arg("birth"), py::arg("death"))
            .def_readwrite("birth", &DgmPoint::birth)
            .def_readwrite("death", &DgmPoint::death)
            .def_readwrite("birth_index", &DgmPoint::birth_index)
            .def_readwrite("death_index", &DgmPoint::death_index)
            .def_property_readonly("persistence", &DgmPoint::persistence)
            .def_property_readonly("index_persistence", &DgmPoint::persistence)
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

    py::class_<IndexDgmPoint>(m, index_dgm_point_name.c_str())
            .def(py::init<size_t, size_t>())
            .def_readwrite("birth", &IndexDgmPoint::birth)
            .def_readwrite("death", &IndexDgmPoint::death)
            .def_property_readonly("persistence", [](const IndexDgmPoint& p)
                 { return std::abs(static_cast<long long>(p.birth) - static_cast<long long>(p.death)); })
            .def("__getitem__", [](const IndexDgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const IndexDgmPoint& p) { return std::hash<IndexDgmPoint>()(p); })
            .def("__eq__", [](const IndexDgmPoint& p, const IndexDgmPoint& q) { return p == q; })
            .def("__repr__", [](const IndexDgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<Diagram>(m, dgm_class_name.c_str())
            .def(py::init<dim_type>())
            .def("in_dimension", [](Diagram& self, dim_type dim, bool as_numpy) -> std::variant<pybind11::array_t<oin_real>, DgmPtVec> {
                      if (as_numpy)
                          return self.get_diagram_in_dimension_as_numpy(dim);
                      else
                          return self.get_diagram_in_dimension(dim);
                    }, "return persistence diagram in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    py::arg("dim"), py::arg("as_numpy") = true)
            .def("index_diagram_in_dimension", [](Diagram& self, dim_type dim, bool as_numpy, bool sorted) -> std::variant<pybind11::array_t<size_t>, IndexDgmPtVec> {
                      if (as_numpy)
                          return self.get_index_diagram_in_dimension_as_numpy(dim, sorted);
                      else
                          return self.get_index_diagram_in_dimension(dim, sorted);
                    }, "return persistence pairing (index diagram) in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    py::arg("dim"), py::arg("as_numpy") = true, py::arg("sorted") =  true)
            .def("__getitem__", &Diagram::get_diagram_in_dimension_as_numpy);
}
