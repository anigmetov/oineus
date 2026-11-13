#include "oineus_persistence_bindings.h"
#include <nanobind/stl/variant.h>

// Registered per Real (double on the top module, float32 in the _f32 submodule).
// `using oin_real = Real` shadows the global alias so the body stays Real-generic
// with no other edits; the one Real-INDEPENDENT class here (IndexDiagramPoint =
// Diagrams<size_t>::Point) is registered only when reg_indep is set.
template<class Real>
void register_oineus_diagram(nb::module_& m, bool reg_indep)
{
    using oin_real = Real;
    using DgmPoint = typename oin::Diagrams<oin_real>::Point;
    using DgmPtVec = typename oin::Diagrams<oin_real>::Dgm;
    using IndexDgmPoint = typename oin::Diagrams<size_t>::Point;
    using IndexDgmPtVec = typename oin::Diagrams<oin_real>::IndexDgm;
    using Diagram = PyOineusDiagrams<oin_real>;
    using DgmPointStateTuple = std::tuple<decltype(DgmPoint::birth),
                                          decltype(DgmPoint::death),
                                          decltype(DgmPoint::birth_index),
                                          decltype(DgmPoint::death_index),
                                          decltype(DgmPoint::birth_index_unsorted),
                                          decltype(DgmPoint::death_index_unsorted),
                                          decltype(DgmPoint::id)>;
    using IndexDgmPointStateTuple = std::tuple<decltype(IndexDgmPoint::birth),
                                               decltype(IndexDgmPoint::death),
                                               decltype(IndexDgmPoint::birth_index),
                                               decltype(IndexDgmPoint::death_index),
                                               decltype(IndexDgmPoint::birth_index_unsorted),
                                               decltype(IndexDgmPoint::death_index_unsorted),
                                               decltype(IndexDgmPoint::id)>;
    using DiagramStateTuple = std::tuple<typename Diagram::State>;

    const std::string dgm_point_name = "DiagramPoint";
    const std::string index_dgm_point_name = "IndexDiagramPoint";
    const std::string dgm_class_name = "Diagrams";

    nb::class_<DgmPoint>(m, dgm_point_name.c_str(),
            "Single persistence-diagram point with birth/death values and critical cell indices.")
            .def(nb::init<oin_real, oin_real>(), nb::arg("birth"), nb::arg("death"))
            .def_rw("birth", &DgmPoint::birth, "Birth filtration value.")
            .def_rw("death", &DgmPoint::death, "Death filtration value (+inf or -inf for essential classes).")
            .def_rw("birth_index", &DgmPoint::birth_index, "Filtration sorted_id of the positive (birth) cell.")
            .def_rw("death_index", &DgmPoint::death_index, "Filtration sorted_id of the negative (death) cell.")
            .def_prop_ro("persistence", &DgmPoint::persistence, "Absolute persistence ``|death - birth|``.")
            .def_prop_ro("index_persistence", &DgmPoint::index_persistence, "Index persistence ``|death_index - birth_index|``.")
            .def("is_diagonal", &DgmPoint::is_diagonal, "Return True if birth == death.")
            .def("is_inf", &DgmPoint::is_inf, "Return True if either coordinate is infinite (essential point).")
            .def("__getitem__", [](const DgmPoint& p, int i) { return p[i]; }, nb::arg("i"),
                 "Return birth (i == 0) or death (i == 1).")
            .def("__hash__", [](const DgmPoint& p) { return std::hash<DgmPoint>()(p); })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            })
            .def("__getstate__", [](const DgmPoint& p) -> DgmPointStateTuple {
                return std::make_tuple(p.birth, p.death, p.birth_index, p.death_index,
                        p.birth_index_unsorted, p.death_index_unsorted, p.id);
            })
            .def("__setstate__", [](DgmPoint& p, const DgmPointStateTuple& t) {
                new (&p) DgmPoint();
                p.birth = std::get<0>(t);
                p.death = std::get<1>(t);
                p.birth_index = std::get<2>(t);
                p.death_index = std::get<3>(t);
                p.birth_index_unsorted = std::get<4>(t);
                p.death_index_unsorted = std::get<5>(t);
                p.id = std::get<6>(t);
            });

    // Diagrams<size_t>::Point is Real-independent -- register it once (default real)
    if (reg_indep)
    nb::class_<IndexDgmPoint>(m, index_dgm_point_name.c_str(),
            "Persistence pairing point expressed in filtration sorted_id indices.")
            .def(nb::init<size_t, size_t>(), nb::arg("birth"), nb::arg("death"))
            .def_rw("birth", &IndexDgmPoint::birth, "Filtration sorted_id of the birth cell.")
            .def_rw("death", &IndexDgmPoint::death, "Filtration sorted_id of the death cell.")
            .def_prop_ro("persistence", [](const IndexDgmPoint& p)
                 { return std::abs(static_cast<long long>(p.birth) - static_cast<long long>(p.death)); },
                 "Index persistence ``|death - birth|``.")
            .def("__getitem__", [](const IndexDgmPoint& p, int i) { return p[i]; }, nb::arg("i"),
                 "Return birth index (i == 0) or death index (i == 1).")
            .def("__hash__", [](const IndexDgmPoint& p) { return std::hash<IndexDgmPoint>()(p); })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__repr__", [](const IndexDgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            })
            .def("__getstate__", [](const IndexDgmPoint& p) -> IndexDgmPointStateTuple {
                return std::make_tuple(p.birth, p.death, p.birth_index, p.death_index,
                        p.birth_index_unsorted, p.death_index_unsorted, p.id);
            })
            .def("__setstate__", [](IndexDgmPoint& p, const IndexDgmPointStateTuple& t) {
                new (&p) IndexDgmPoint();
                p.birth = std::get<0>(t);
                p.death = std::get<1>(t);
                p.birth_index = std::get<2>(t);
                p.death_index = std::get<3>(t);
                p.birth_index_unsorted = std::get<4>(t);
                p.death_index_unsorted = std::get<5>(t);
                p.id = std::get<6>(t);
            });

    nb::class_<Diagram>(m, dgm_class_name.c_str(),
            "Container of persistence diagrams across homology dimensions 0..n_dims-1.")
            .def(nb::init<dim_type>(), nb::arg("max_dim"))
            .def("in_dimension", [](Diagram& self, dim_type dim, bool as_numpy) -> std::variant<nb::ndarray<oin_real, nb::numpy>, DgmPtVec> {
                      try {
                          if (as_numpy)
                              return self.get_diagram_in_dimension_as_numpy(dim);
                          else
                              return self.get_diagram_in_dimension(dim);
                      } catch (const std::out_of_range&) {
                          throw nb::index_error("Diagram dimension out of range");
                      }
                    }, "Return the persistence diagram in dimension dim: as an (n, 2) NumPy array "
                       "when as_numpy is True (the default), or as a list of DiagramPoint objects "
                       "(carrying birth_index and death_index) when as_numpy is False.",
                    nb::arg("dim"), nb::arg("as_numpy") = true)
            .def("index_diagram_in_dimension", [](Diagram& self, dim_type dim, bool as_numpy) -> std::variant<nb::ndarray<size_t, nb::numpy>, IndexDgmPtVec> {
                      try {
                          if (as_numpy)
                              return self.get_index_diagram_in_dimension_as_numpy(dim);
                          else
                              return self.get_index_diagram_in_dimension(dim);
                      } catch (const std::out_of_range&) {
                          throw nb::index_error("Diagram dimension out of range");
                      }
                    }, "Return the persistence index pairing (including zero-persistence pairs) in "
                       "dimension dim: as an (n, 2) size_t NumPy array of sorted_ids when as_numpy is "
                       "True (the default), or as a list of IndexDiagramPoint objects when False.",
                    nb::arg("dim"), nb::arg("as_numpy") = true)
            .def("__getitem__", [](Diagram& self, dim_type dim) {
                try {
                    return self.get_diagram_in_dimension_as_numpy(dim);
                } catch (const std::out_of_range&) {
                    throw nb::index_error("Diagram dimension out of range");
                }
            }, nb::arg("dim"), "Shorthand for in_dimension(dim, as_numpy=True).")
            .def("__len__", &Diagram::n_dims, "Number of stored homology dimensions.")
            .def("pad_to_dim", &Diagram::pad_to_dim, nb::arg("new_top_dim"),
                 "Extend diagrams to dimensions [0..new_top_dim] by appending empty diagrams as needed.")
            .def("trim_to_dim", &Diagram::trim_to_dim, nb::arg("max_dim"),
                 "Trim diagrams to dimensions [0..max_dim].")
            .def("__repr__", [](const Diagram& self) {
                const auto& dims = self.data().diagram_in_dimension_;
                std::stringstream ss;
                ss << "Diagrams(n_dims=" << dims.size() << ", sizes=[";
                for (size_t d = 0; d < dims.size(); ++d) {
                    if (d) ss << ", ";
                    ss << dims[d].size();
                }
                ss << "])";
                return ss.str();
            })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("__getstate__", [](const Diagram& d) -> DiagramStateTuple {
                return std::make_tuple(d.state());
            })
            .def("__setstate__", [](Diagram& d, const DiagramStateTuple& t) {
                new (&d) Diagram();
                d.set_state(std::get<0>(t));
            })
            ;
}

// double pass on the top module (registers the Real-independent IndexDiagramPoint too)
void init_oineus_diagram(nb::module_& m) { register_oineus_diagram<double>(m, true); }

// float pass is compiled here; the driver calls it into the _f32 submodule
template void register_oineus_diagram<float>(nb::module_&, bool);
