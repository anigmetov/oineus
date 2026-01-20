#include "oineus_persistence_bindings.h"

void init_oineus_cells(nb::module_& m)
{
    using Simplex = oin::Simplex<oin_int>;
    using SimplexValue = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexValue = oin::CellWithValue<ProdSimplex, oin_real>;

    // Simplex and product of two simplices without value, vertices only
    const std::string pure_simplex_class_name = "CombinatorialSimplex";
    const std::string pure_prod_simplex_class_name = "CombinatorialProdSimplex";
    const std::string pure_cube_class_name = "CombinatorialCube";

    const std::string simplex_class_name = "Simplex";
    const std::string prod_simplex_class_name = "ProdSimplex";
    const std::string domain_class_name = "Grid";
    const std::string cube_class_name = "CubeValue";

    using SimplexStateTuple = std::tuple<decltype(Simplex::id_),
                                         decltype(Simplex::uid_),
                                         decltype(Simplex::vertices_)
                                         >;

    using SimplexValueStateTuple = std::tuple<decltype(SimplexValue::value_),
                                              decltype(SimplexValue::sorted_id_),
                                              decltype(SimplexValue::cell_)
                                             >;

    using ProdSimplexStateTuple = std::tuple<decltype(ProdSimplex::id_),
                                             decltype(ProdSimplex::cell_1_),
                                             decltype(ProdSimplex::cell_2_)
                                            >;

    using ProdSimplexValueStateTuple = std::tuple<decltype(ProdSimplexValue::value_),
                                                  decltype(ProdSimplexValue::sorted_id_),
                                                  decltype(ProdSimplexValue::cell_)
                                                 >;

    nb::class_<Simplex>(m, pure_simplex_class_name.c_str())
        .def(nb::init<const Simplex::IdxVector&>(), nb::arg("vertices"))
        .def(nb::init<oin_int, const Simplex::IdxVector&>(), nb::arg("id"), nb::arg("vertices"))
        .def("__iter__", [](Simplex& sigma) { return nb::make_iterator(nb::type<Simplex>(), "vertices_iterator", sigma.vertices_.begin(), sigma.vertices_.end()); }, nb::keep_alive<0, 1>())
        .def("__getitem__", [](Simplex& sigma, size_t i) { return sigma.vertices_[i]; })
        .def_prop_rw("id", &Simplex::get_id, &Simplex::set_id)
        .def_prop_ro("vertices", &Simplex::get_vertices)
        .def_prop_ro("uid", &Simplex::get_uid)
        .def_prop_ro("dim", &Simplex::dim)
        .def("boundary", &Simplex::boundary)
        .def("join", [](const Simplex& sigma, oin_int new_vertex, oin_int new_id) {
                  return sigma.join(new_id, new_vertex);
                },
                nb::arg("new_vertex"),
                nb::arg("new_id") = Simplex::k_invalid_id)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::hash(nb::self))
        .def("__getstate__", [](const Simplex& sigma) { return std::make_tuple(sigma.id_, sigma.uid_, sigma.vertices_); })
        .def("__setstate__", [](Simplex& sigma, const SimplexStateTuple& state) {
            new (&sigma) Simplex();
            sigma.id_ = std::get<0>(state);
            sigma.uid_ = std::get<1>(state);
            sigma.vertices_ = std::get<2>(state);
         })
        .def("__repr__", [](const Simplex& sigma) {
          std::stringstream ss;
          ss << sigma;
          return ss.str();
        });

    nb::class_<ProdSimplex>(m, pure_prod_simplex_class_name.c_str())
        .def(nb::init<const Simplex::IdxVector&, const Simplex::IdxVector&>(), nb::arg("vertices_1"), nb::arg("vertices_2"))
        .def_prop_rw("id", &ProdSimplex::get_id, &ProdSimplex::set_id)
        .def_prop_ro("factor_1", &ProdSimplex::get_factor_1)
        .def_prop_ro("factor_2", &ProdSimplex::get_factor_2)
        .def_prop_ro("uid", &ProdSimplex::get_uid)
        .def_prop_ro("dim", &ProdSimplex::dim)
        .def("boundary", &ProdSimplex::boundary)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::hash(nb::self))
        .def("__getstate__", [](const ProdSimplex& sigma) ->ProdSimplexStateTuple { return std::make_tuple(sigma.id_, sigma.get_factor_1(), sigma.get_factor_2()); })
        .def("__setstate__", [](ProdSimplex& sigma, const ProdSimplexStateTuple& state) {
            new (&sigma) ProdSimplex();
            sigma.id_ = std::get<0>(state);
            sigma.cell_1_ = std::get<1>(state);
            sigma.cell_2_ = std::get<2>(state);
         })
        .def("__repr__", [](const ProdSimplex& sigma) {
          std::stringstream ss;
          ss << sigma;
          return ss.str();
        });

    nb::class_<SimplexValue>(m, simplex_class_name.c_str())
            .def(nb::init<const Simplex::IdxVector&, oin_real>(),
                    nb::arg("vertices"),
                    nb::arg("value")=0.0)
            .def("__init__", [](SimplexValue * p, oin_int id, const Simplex::IdxVector& vs, oin_real value) {
                    new (p) SimplexValue(Simplex(id, vs), value);
                }, nb::arg("id"), nb::arg("vertices"), nb::arg("value"))
            .def("__iter__", [](SimplexValue& sigma) { return nb::make_iterator(nb::type<SimplexValue>(), "vertex_iterator", sigma.cell_.vertices_.begin(), sigma.cell_.vertices_.end()); }, nb::keep_alive<0, 1>())
            .def("__getitem__", [](SimplexValue& sigma, size_t i) { return sigma.cell_.vertices_[i]; })
            .def_prop_rw("id", &SimplexValue::get_id, &SimplexValue::set_id)
            .def_rw("sorted_id", &SimplexValue::sorted_id_)
            .def_prop_ro("vertices", &SimplexValue::get_vertices<Simplex>, "simplex vertices")
            .def_prop_ro("uid", &SimplexValue::get_uid)
            .def_rw("value", &SimplexValue::value_)
            .def_prop_ro("dim", &SimplexValue::dim)
            .def("boundary", &SimplexValue::boundary)
            .def_prop_ro("combinatorial_simplex", &SimplexValue::get_cell)
            .def_prop_ro("combinatorial_cell", &SimplexValue::get_cell)
            .def("join", [](const SimplexValue& sigma, oin_int new_vertex, oin_real value, oin_int new_id) {
                      return sigma.join(new_id, new_vertex, value);
                    },
                    nb::arg("new_vertex"),
                    nb::arg("value"),
                    nb::arg("new_id") = SimplexValue::k_invalid_id)
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def(nb::hash(nb::self))
            .def("__getstate__", [](const SimplexValue& sigma) { return std::make_tuple(sigma.value_, sigma.sorted_id_, sigma.cell_); })
            .def("__setstate__", [](SimplexValue& sigma, const SimplexValueStateTuple& state) {
                new (&sigma) SimplexValue();
                sigma.value_ = std::get<0>(state);
                sigma.sorted_id_ = std::get<1>(state);
                sigma.cell_ = std::get<2>(state);
             })
            .def("__repr__", [](const SimplexValue& sigma) {
                  std::stringstream ss;
                  ss << sigma;
                  return ss.str();
                });

    nb::class_<ProdSimplexValue>(m, prod_simplex_class_name.c_str())
             .def("__init__", [](ProdSimplexValue* p, const Simplex::IdxVector& vertices_1, const Simplex::IdxVector& vertices_2, oin_real value) {
                      new (p) ProdSimplexValue(ProdSimplex(Simplex(vertices_1), Simplex(vertices_2)), value);
                    },
                    nb::arg("vertices_1"),
                    nb::arg("vertices_2"),
                    nb::arg("value"))
            .def("__init__", [](ProdSimplexValue* p, const SimplexValue& sigma, const SimplexValue& tau, oin_real value) {
                      new (p) ProdSimplexValue(ProdSimplex(sigma.get_cell(), tau.get_cell()), value);
                    },
                    nb::arg("cell_1"),
                    nb::arg("cell_2"),
                    nb::arg("value"))
            .def("__init__", [](ProdSimplexValue* p, const Simplex& sigma, const Simplex& tau, oin_real value) {
                      new (p) ProdSimplexValue(ProdSimplex(sigma, tau), value);
                    },
                    nb::arg("cell_1"),
                    nb::arg("cell_2"),
                    nb::arg("value"))
            .def_prop_rw("id", &ProdSimplexValue::get_id, &ProdSimplexValue::set_id)
            .def_rw("sorted_id", &ProdSimplexValue::sorted_id_)
            .def_prop_ro("factor_1", &ProdSimplexValue::get_factor_1)
            .def_prop_ro("factor_2", &ProdSimplexValue::get_factor_2)
            .def_prop_ro("cell_1", &ProdSimplexValue::get_factor_1)
            .def_prop_ro("cell_2", &ProdSimplexValue::get_factor_2)
            .def_prop_ro("uid", &ProdSimplexValue::get_uid)
            .def_rw("value", &ProdSimplexValue::value_)
            .def_prop_ro("dim", &ProdSimplexValue::dim)
            .def("boundary", &ProdSimplexValue::boundary)
            .def_prop_ro("combinatorial_cell", &ProdSimplexValue::get_cell)
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def(nb::hash(nb::self))
            .def("__getstate__", [](const ProdSimplexValue& sigma) -> ProdSimplexValueStateTuple { return std::make_tuple(sigma.value_, sigma.sorted_id_, sigma.cell_); })
            .def("__setstate__", [](ProdSimplexValue& sigma, const ProdSimplexValueStateTuple& state) {
                new (&sigma) ProdSimplexValue();
                sigma.value_ = std::get<0>(state);
                sigma.sorted_id_ = std::get<1>(state);
                sigma.cell_ = std::get<2>(state);
             })
            .def("__repr__", [](const ProdSimplexValue& sigma) {
                      std::stringstream ss;
                      ss << sigma;
                      return ss.str();
                    });

    using GridDomain_1D = oin::GridDomain<oin_int, 1>;
    using GridDomain_2D = oin::GridDomain<oin_int, 2>;
    using GridDomain_3D = oin::GridDomain<oin_int, 3>;

    using Grid_1D = oin::Grid<oin_int, oin_real, 1>;
    using Grid_2D = oin::Grid<oin_int, oin_real, 2>;
    using Grid_3D = oin::Grid<oin_int, oin_real, 3>;

    using Cube_1D = oin::Cube<oin_int, 1>;
    using Cube_2D = oin::Cube<oin_int, 2>;
    using Cube_3D = oin::Cube<oin_int, 3>;


    using CubeValue_1D = oin::CellWithValue<oin::Cube<oin_int, 1>, oin_real>;
    using CubeValue_2D = oin::CellWithValue<oin::Cube<oin_int, 2>, oin_real>;
    using CubeValue_3D = oin::CellWithValue<oin::Cube<oin_int, 3>, oin_real>;

    // ============ GridDomain bindings ============
    #define BIND_GRID_DOMAIN(DIM) \
        using GridDomain_##DIM##DStateTuple = std::tuple<decltype(GridDomain_##DIM##D::dims_), \
                                                        decltype(GridDomain_##DIM##D::strides_), \
                                                        decltype(GridDomain_##DIM##D::c_order_), \
                                                        decltype(GridDomain_##DIM##D::wrap_) \
                                                        >; \
        nb::class_<GridDomain_##DIM##D>(m, "GridDomain_" #DIM "D", #DIM "D grid domain") \
            .def("__init__", ([](GridDomain_##DIM##D* p, nb::args args) { \
                    if (args.size() != DIM) \
                        throw std::runtime_error("Expected " #DIM " arguments"); \
                    GridDomain_##DIM##D::GridPoint shape; \
                    for (size_t i = 0; i < DIM; ++i) \
                        shape[i] = nb::cast<int>(args[i]); \
                    new (p) GridDomain_##DIM##D(shape, false); \
                })) \
            .def_prop_ro("shape", [](const GridDomain_##DIM##D& g) { return g.shape(); }) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def(nb::hash(nb::self)) \
            .def("__getstate__", [](const GridDomain_##DIM##D& g) { return std::make_tuple(g.dims_, g.strides_, g.c_order_, g.wrap_); }) \
            .def("__setstate__", [](GridDomain_##DIM##D& g, const GridDomain_##DIM##DStateTuple& state) { \
                new (&g) GridDomain_##DIM##D(); \
                g.dims_ = std::get<0>(state); \
                g.strides_ = std::get<1>(state); \
                g.c_order_ = std::get<2>(state); \
                g.wrap_ = std::get<3>(state); \
             }) \

    BIND_GRID_DOMAIN(1);
    BIND_GRID_DOMAIN(2);
    BIND_GRID_DOMAIN(3);

    #undef BIND_GRID_DOMAIN

    // ============ Grid bindings ============
    #define BIND_GRID(DIM) \
        nb::class_<Grid_##DIM##D>(m, "Grid_" #DIM "D", #DIM "D grid with data") \
            .def("__init__", ([](Grid_##DIM##D * p, nb::ndarray<oin_real, nb::c_contig, nb::device::cpu> data, bool wrap, std::string values_on) \
                { \
                    if (data.ndim() != DIM) \
                        throw std::runtime_error("Array must be " #DIM "D"); \
                    oin_real* pdata {static_cast<oin_real*>(data.data())}; \
                    Grid_##DIM##D::GridPoint shape; \
                    for (size_t i = 0; i < DIM; ++i) \
                        shape[i] = data.shape(i); \
                    Grid_##DIM##D::DataLocation data_loc; \
                    if (values_on == "cells") \
                        data_loc = Grid_##DIM##D::DataLocation::CELL ; \
                    else if (values_on == "vertices") \
                        data_loc = Grid_##DIM##D::DataLocation::VERTEX ; \
                    else \
                        throw std::runtime_error("values_on must be either 'vertices' or 'cells'"); \
                    new (p) Grid_##DIM##D(shape, wrap, pdata, data_loc); \
                }), nb::arg("data"), nb::arg("wrap") = false, nb::arg("values_on") = "vertices") \
            /*.def_prop_ro("shape", [](const Grid_##DIM##D& g) { return g.domain().shape(); }) */ \
            .def_prop_ro("data_location", &Grid_##DIM##D::data_location_as_string) \
            .def("cube_filtration", &Grid_##DIM##D::cube_filtration, \
                  nb::arg("max_dim") = DIM, nb::arg("negate") = false, nb::arg("n_threads") = 1) \
            .def("cube_filtration_and_critical_indices", &Grid_##DIM##D::cube_filtration_and_critical_indices, \
                  nb::arg("max_dim") = DIM, nb::arg("negate") = false, nb::arg("n_threads") = 1) \
            .def("freudenthal_filtration", &Grid_##DIM##D::freudenthal_filtration, \
                  nb::arg("max_dim") = DIM, nb::arg("negate") = false, nb::arg("n_threads") = 1) \
            .def("freudenthal_filtration_and_critical_vertices", &Grid_##DIM##D::freudenthal_filtration_and_critical_vertices, \
                  nb::arg("max_dim") = DIM, nb::arg("negate") = false, nb::arg("n_threads") = 1) \


    BIND_GRID(1);
    BIND_GRID(2);
    BIND_GRID(3);

    #undef BIND_GRID

    // ============ Cube bindings ============
    #define BIND_CUBE(DIM) \
        using Cube_##DIM##DStateTuple = std::tuple<decltype(Cube_##DIM##D::id_), \
                                                   decltype(Cube_##DIM##D::user_id_), \
                                                   decltype(Cube_##DIM##D::global_domain_) \
                                                  >; \
        nb::class_<Cube_##DIM##D>(m, "CombinatorialCube_" #DIM "D") \
            .def("__init__", [](Cube_##DIM##D * p, const GridDomain_##DIM##D& g, oin_int x) { \
                    new (p) Cube_##DIM##D(x, g); \
                }, nb::arg("domain"), nb::arg("x")) \
            .def(nb::init<const Cube_##DIM##D::Point&, const std::vector<oin_int>&, const GridDomain_##DIM##D&>(), \
                 nb::arg("anchor_vertex"), nb::arg("spanning_dims"), nb::arg("domain")) \
            .def_prop_ro("dim", &Cube_##DIM##D::dim) \
            .def_prop_ro("uid", &Cube_##DIM##D::get_uid, "Get UID of a cube") \
            .def_prop_ro("vertices", &Cube_##DIM##D::get_vertices, "Get all vertices of a cube") \
            .def_prop_ro("anchor_vertex", &Cube_##DIM##D::anchor_vertex, "Get anchor vertex of a cube") \
            .def_prop_rw("id", &Cube_##DIM##D::get_id, &Cube_##DIM##D::set_id, "User ID of a cube") \
            .def_prop_ro("domain", &Cube_##DIM##D::global_domain) \
            .def("boundary", &Cube_##DIM##D::boundary_cubes, "boundary of a cube") \
            .def("coboundary", &Cube_##DIM##D::coboundary_cubes, "coboundary of a cube") \
            .def("top_cofaces", &Cube_##DIM##D::top_cofaces_cubes, "top cofaces of a cube") \
            .def("__repr__", &Cube_##DIM##D::pretty_print) \
            .def("__str__", &Cube_##DIM##D::pretty_print) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def(nb::hash(nb::self)) \
            .def("__getstate__", [](const Cube_##DIM##D& c) -> Cube_##DIM##DStateTuple { return std::make_tuple(c.id_, c.user_id_, c.global_domain_); }) \
            .def("__setstate__", [](Cube_##DIM##D& c, const Cube_##DIM##DStateTuple& state) { \
                    new (&c) Cube_##DIM##D(); \
                    c.id_ = std::get<0>(state); \
                    c.user_id_ = std::get<1>(state); \
                    c.global_domain_ = std::get<2>(state); \
                }) \

    BIND_CUBE(1);
    BIND_CUBE(2);
    BIND_CUBE(3);

    #undef BIND_CUBE

    #define BIND_CUBE_VALUE(DIM) \
         using CubeValue_##DIM##DStateTuple = std::tuple<decltype(CubeValue_##DIM##D::value_), \
                                                  decltype(CubeValue_##DIM##D::sorted_id_), \
                                                  decltype(CubeValue_##DIM##D::cell_) \
                                                 >; \
        nb::class_<CubeValue_##DIM##D>(m, "Cube_" #DIM "D") \
            .def("__init__", [](CubeValue_##DIM##D * p, const GridDomain_##DIM##D& g, oin_int x, oin_real value) { \
                    new (p) CubeValue_##DIM##D(Cube_##DIM##D(x, g), value); \
                }, nb::arg("domain"), nb::arg("x"), nb::arg("value")) \
            .def("__init__", [](CubeValue_##DIM##D * p, const Cube_##DIM##D::Point& anchor, const std::vector<oin_int>& spanning_dims, const GridDomain_##DIM##D& domain, oin_real value) { \
                    new (p) CubeValue_##DIM##D(Cube_##DIM##D(anchor, spanning_dims, domain), value); \
                }, nb::arg("anchor_vertex"), nb::arg("spanning_dims"), nb::arg("domain"), nb::arg("value")) \
            .def_prop_ro("dim", &CubeValue_##DIM##D::dim) \
            .def_prop_ro("uid", &CubeValue_##DIM##D::get_uid, "Get UID of a cube") \
            .def_rw("value", &CubeValue_##DIM##D::value_) \
            .def_prop_ro("vertices", &CubeValue_##DIM##D::get_vertices<Cube_##DIM##D>, "Get all vertices of a cube") \
            .def("boundary", &CubeValue_##DIM##D::boundary, "boundary of a cube") \
            .def("coboundary", &CubeValue_##DIM##D::coboundary_cubes<Cube_##DIM##D>, "coboundary of a cube") \
            .def("top_cofaces", &CubeValue_##DIM##D::top_cofaces<Cube_##DIM##D>, "top-dim cofaces of a cube") \
            .def("__repr__", &CubeValue_##DIM##D::pretty_print) \
            .def("__str__", &CubeValue_##DIM##D::pretty_print) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def(nb::hash(nb::self)) \
            .def("__getstate__", [](const CubeValue_##DIM##D& c) -> CubeValue_##DIM##DStateTuple { return std::make_tuple(c.value_, c.sorted_id_, c.cell_); }) \
            .def("__setstate__", [](CubeValue_##DIM##D& c, const CubeValue_##DIM##DStateTuple & state) { \
                new (&c) ProdSimplexValue(); \
                c.value_ = std::get<0>(state); \
                c.sorted_id_ = std::get<1>(state); \
                c.cell_ = std::get<2>(state); \
             }) \

    BIND_CUBE_VALUE(1);
    BIND_CUBE_VALUE(2);
    BIND_CUBE_VALUE(3);

    #undef BIND_CUBE_VALUE
}
