#include "oineus_persistence_bindings.h"

void init_oineus_common_diagram(py::module& m)
{
    using namespace pybind11::literals;

    using oin::VREdge;
    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;

    using DgmPoint_int = typename oin::DgmPoint<oin_int>;
    using DgmPointSizet = typename oin::DgmPoint<size_t>;

    py::class_<DgmPoint_int>(m, "DgmPoint_int")
            .def(py::init<oin_int, oin_int>())
            .def_readwrite("birth", &DgmPoint_int::birth)
            .def_readwrite("death", &DgmPoint_int::death)
            .def("__getitem__", [](const DgmPoint_int& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPoint_int& p) { return std::hash<DgmPoint_int>()(p); })
            .def("__eq__", [](const DgmPoint_int& p, const DgmPoint_int& q) { return p == q; })
            .def("__repr__", [](const DgmPoint_int& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<DgmPointSizet>(m, "DgmPoint_Sizet")
            .def(py::init<size_t, size_t>())
            .def_readwrite("birth", &DgmPointSizet::birth)
            .def_readwrite("death", &DgmPointSizet::death)
            .def("__getitem__", [](const DgmPointSizet& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPointSizet& p) { return std::hash<DgmPointSizet>()(p); })
            .def("__eq__", [](const DgmPointSizet& p, const DgmPointSizet& q) { return p == q; })
            .def("__repr__", [](const DgmPointSizet& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });
}
