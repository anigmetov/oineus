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

    nb::class_<Simplex>(m, pure_simplex_class_name.c_str())
        .def(nb::init<const Simplex::IdxVector&>(), nb::arg("vertices"))
        .def(nb::init<oin_int, const Simplex::IdxVector&>(), nb::arg("id"), nb::arg("vertices"))
        .def("__iter__", [](Simplex& sigma) { return nb::make_iterator(nb::type<Simplex>(), "vertices_iterator", sigma.vertices_.begin(), sigma.vertices_.end()); }, nb::keep_alive<0, 1>())
        .def("__getitem__", [](Simplex& sigma, size_t i) { return sigma.vertices_[i]; })
        .def_prop_rw("id", &Simplex::get_id, &Simplex::set_id)
        .def_ro("vertices", &Simplex::vertices_)
        .def_prop_ro("uid", &Simplex::get_uid)
        .def("dim", &Simplex::dim)
        .def("boundary", &Simplex::boundary)
        .def("join", [](const Simplex& sigma, oin_int new_vertex, oin_int new_id) {
                  return sigma.join(new_id, new_vertex);
                },
                nb::arg("new_vertex"),
                nb::arg("new_id") = Simplex::k_invalid_id)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::hash(nb::self))
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
        .def("dim", &ProdSimplex::dim)
        .def("boundary", &ProdSimplex::boundary)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::hash(nb::self))
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
            .def_prop_ro("vertices", [](const SimplexValue& sigma) { return sigma.cell_.vertices_; })
            .def_prop_ro("uid", &SimplexValue::get_uid)
            .def_rw("value", &SimplexValue::value_)
            .def("dim", &SimplexValue::dim)
            .def("boundary", &SimplexValue::boundary)
            .def("combinatorial_simplex", &SimplexValue::get_cell)
            .def("combinatorial_cell", &SimplexValue::get_cell)
            .def("join", [](const SimplexValue& sigma, oin_int new_vertex, oin_real value, oin_int new_id) {
                      return sigma.join(new_id, new_vertex, value);
                    },
                    nb::arg("new_vertex"),
                    nb::arg("value"),
                    nb::arg("new_id") = SimplexValue::k_invalid_id)
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def(nb::hash(nb::self))
            .def("__repr__", [](const SimplexValue& sigma) {
                  std::stringstream ss;
                  ss << sigma;
                  return ss.str();
                });

    nb::class_<ProdSimplexValue>(m, prod_simplex_class_name.c_str())
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
            .def_prop_ro("cell_1", &ProdSimplexValue::get_factor_1)
            .def_prop_ro("cell_2", &ProdSimplexValue::get_factor_2)
            .def_prop_ro("uid", &ProdSimplexValue::get_uid)
            .def_rw("value", &ProdSimplexValue::value_)
            .def("dim", &ProdSimplexValue::dim)
            .def("boundary", &ProdSimplexValue::boundary)
            .def("combinatorial_cell", &ProdSimplexValue::get_cell)
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def(nb::hash(nb::self))
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

    // ============ GridDomain bindings ============
    #define BIND_GRID_DOMAIN(DIM) \
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
        nb::class_<Cube_##DIM##D>(m, "Cube_" #DIM "D") \
            .def("__init__", [](Cube_##DIM##D * p, const GridDomain_##DIM##D& g, oin_int x) { \
                    new (p) Cube_##DIM##D(x, g); \
                }, nb::arg("domain"), nb::arg("x")) \
            .def(nb::init<const Cube_##DIM##D::Point&, const std::vector<oin_int>&, const GridDomain_##DIM##D&>(), \
                 nb::arg("anchor_vertex"), nb::arg("spanning_dims"), nb::arg("domain")) \
            .def_prop_ro("dim", &Cube_##DIM##D::dim) \
            .def_prop_ro("uid", &Cube_##DIM##D::get_uid, "Get UID of a cube") \
            .def_prop_ro("vertices", &Cube_##DIM##D::vertices, "Get all vertices of a cube") \
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


    BIND_CUBE(1);
    BIND_CUBE(2);
    BIND_CUBE(3);

    #undef BIND_CUBE
}