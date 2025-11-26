#include "pybind11/operators.h"
#include "oineus_persistence_bindings.h"

void init_oineus_cells(py::module& m)
{
    using namespace pybind11::literals;

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

    py::class_<Simplex>(m, pure_simplex_class_name.c_str())
        .def(py::init([](Simplex::IdxVector vs) -> Simplex { return Simplex(vs, true); }), py::arg("vertices"))
        .def(py::init([](oin_int id, Simplex::IdxVector vs) -> Simplex { return Simplex(id, vs, true); }), py::arg("id"), py::arg("vertices"))
        .def("__iter__", [](Simplex& sigma) { return py::make_iterator(sigma.vertices_.begin(), sigma.vertices_.end()); }, py::keep_alive<0, 1>())
        .def("__getitem__", [](Simplex& sigma, size_t i) { return sigma.vertices_[i]; })
        .def_property("id", &Simplex::get_id, &Simplex::set_id)
        .def_readonly("vertices", &Simplex::vertices_)
        .def_property_readonly("uid", &Simplex::get_uid)
        .def("dim", &Simplex::dim)
        .def("boundary", &Simplex::boundary)
        .def("join", [](const Simplex& sigma, oin_int new_vertex, oin_int new_id) {
                  return sigma.join(new_id, new_vertex);
                },
                py::arg("new_vertex"),
                py::arg("new_id") = Simplex::k_invalid_id)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::hash(py::self))
        .def("__repr__", [](const Simplex& sigma) {
          std::stringstream ss;
          ss << sigma;
          return ss.str();
        });

    py::class_<ProdSimplex>(m, pure_prod_simplex_class_name.c_str())
        .def(py::init([](Simplex::IdxVector vs1, Simplex::IdxVector vs2) -> ProdSimplex
            { return ProdSimplex(Simplex(vs1, true), Simplex(vs2, true)); }), py::arg("vertices_1"), py::arg("vertices_2"))
        .def_property("id", &ProdSimplex::get_id, &ProdSimplex::set_id)
        .def_property_readonly("factor_1", &ProdSimplex::get_factor_1)
        .def_property_readonly("factor_2", &ProdSimplex::get_factor_2)
        .def_property_readonly("uid", &ProdSimplex::get_uid)
        .def("dim", &ProdSimplex::dim)
        .def("boundary", &ProdSimplex::boundary)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::hash(py::self))
        .def("__repr__", [](const ProdSimplex& sigma) {
          std::stringstream ss;
          ss << sigma;
          return ss.str();
        });

    py::class_<SimplexValue>(m, simplex_class_name.c_str())
            .def(py::init([](Simplex::IdxVector vs, oin_real value) -> SimplexValue {
                      return SimplexValue({vs}, value);
                    }),
                    py::arg("vertices"),
                    py::arg("value")=0.0)
            .def(py::init([](oin_int id, Simplex::IdxVector vs, oin_real value) -> SimplexValue { return SimplexValue({id, vs}, value); }), py::arg("id"), py::arg("vertices"), py::arg("value"))
            .def("__iter__", [](SimplexValue& sigma) { return py::make_iterator(sigma.cell_.vertices_.begin(), sigma.cell_.vertices_.end()); }, py::keep_alive<0, 1>())
            .def("__getitem__", [](SimplexValue& sigma, size_t i) { return sigma.cell_.vertices_[i]; })
            .def_property("id", &SimplexValue::get_id, &SimplexValue::set_id)
            .def_readwrite("sorted_id", &SimplexValue::sorted_id_)
            .def_property_readonly("vertices", [](const SimplexValue& sigma) { return sigma.cell_.vertices_; })
            .def_property_readonly("uid", &SimplexValue::get_uid)
            .def_readwrite("value", &SimplexValue::value_)
            .def("dim", &SimplexValue::dim)
            .def("boundary", &SimplexValue::boundary)
            .def("combinatorial_simplex", &SimplexValue::get_cell)
            .def("combinatorial_cell", &SimplexValue::get_cell)
            .def("join", [](const SimplexValue& sigma, oin_int new_vertex, oin_real value, oin_int new_id) {
                      return sigma.join(new_id, new_vertex, value);
                    },
                    py::arg("new_vertex"),
                    py::arg("value"),
                    py::arg("new_id") = SimplexValue::k_invalid_id)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::hash(py::self))
            .def("__repr__", [](const SimplexValue& sigma) {
                  std::stringstream ss;
                  ss << sigma;
                  return ss.str();
                });

    py::class_<ProdSimplexValue>(m, prod_simplex_class_name.c_str())
            .def(py::init([](const SimplexValue& sigma, const SimplexValue& tau, oin_real value) -> ProdSimplexValue {
                      return ProdSimplexValue(ProdSimplex(sigma.get_cell(), tau.get_cell()), value);
                    }),
                    py::arg("cell_1"),
                    py::arg("cell_2"),
                    py::arg("value"))
            .def(py::init([](const Simplex& sigma, const Simplex& tau, oin_real value) -> ProdSimplexValue {
                      return ProdSimplexValue(ProdSimplex(sigma, tau), value);
                    }),
                    py::arg("cell_1"),
                    py::arg("cell_2"),
                    py::arg("value"))
            .def_property("id", &ProdSimplexValue::get_id, &ProdSimplexValue::set_id)
            .def_readwrite("sorted_id", &ProdSimplexValue::sorted_id_)
            .def_property_readonly("cell_1", &ProdSimplexValue::get_factor_1)
            .def_property_readonly("cell_2", &ProdSimplexValue::get_factor_2)
            .def_property_readonly("uid", &ProdSimplexValue::get_uid)
            .def_readwrite("value", &ProdSimplexValue::value_)
            .def("dim", &ProdSimplexValue::dim)
            .def("boundary", &ProdSimplexValue::boundary)
            .def("combinatorial_cell", &ProdSimplexValue::get_cell)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::hash(py::self))
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
        py::class_<GridDomain_##DIM##D>(m, "GridDomain_" #DIM "D", #DIM "D grid domain") \
            .def(py::init([](py::args args) -> GridDomain_##DIM##D { \
                    if (args.size() != DIM) \
                        throw std::runtime_error("Expected " #DIM " arguments"); \
                    GridDomain_##DIM##D::GridPoint shape; \
                    for (size_t i = 0; i < DIM; ++i) \
                        shape[i] = args[i].cast<int>(); \
                    return GridDomain_##DIM##D(shape, false); \
                })) \
            .def_property_readonly("shape", [](const GridDomain_##DIM##D& g) { return g.shape(); }) \
            .def(py::self == py::self) \
            .def(py::self != py::self) \
            .def(py::hash(py::self)) \


    BIND_GRID_DOMAIN(1);
    BIND_GRID_DOMAIN(2);
    BIND_GRID_DOMAIN(3);

    #undef BIND_GRID_DOMAIN

    // ============ Grid bindings ============
    #define BIND_GRID(DIM) \
        py::class_<Grid_##DIM##D>(m, "Grid_" #DIM "D", #DIM "D grid with data") \
            .def(py::init([](py::array_t<oin_real, py::array::c_style | py::array::forcecast> data, bool wrap) -> Grid_##DIM##D \
                { \
                    py::buffer_info data_buf = data.request(); \
                    if (data_buf.ndim != DIM) \
                        throw std::runtime_error("Array must be " #DIM "D"); \
                    oin_real* pdata {static_cast<oin_real*>(data_buf.ptr)}; \
                    Grid_##DIM##D::GridPoint shape; \
                    for (size_t i = 0; i < DIM; ++i) \
                        shape[i] = data_buf.shape[i]; \
                    return Grid_##DIM##D(shape, wrap, pdata); \
                }), py::arg("data"), py::arg("wrap") = false) \
            .def_property_readonly("shape", [](const Grid_##DIM##D& g) { return g.domain().shape(); }) \
            .def("cube_filtration", [](const Grid_##DIM##D& g, size_t top_d, bool negate, bool cell_centric, int n_threads) { return g.cube_filtration(top_d, negate, cell_centric, n_threads); }, \
                  py::arg("max_dim"), py::arg("negate") = false, py::arg("cell_centric") = false, py::arg("n_threads") = 1) \


    BIND_GRID(1);
    BIND_GRID(2);
    BIND_GRID(3);

    #undef BIND_GRID

    // ============ Cube bindings ============
    #define BIND_CUBE(DIM) \
        py::class_<Cube_##DIM##D>(m, "Cube_" #DIM "D") \
            .def(py::init([](const GridDomain_##DIM##D& g, oin_int x) -> Cube_##DIM##D { \
                    return Cube_##DIM##D(x, g); \
                }), py::arg("domain"), py::arg("x")) \
            .def(py::init<const Cube_##DIM##D::Point&, const std::vector<oin_int>&, const GridDomain_##DIM##D&>(), \
                 py::arg("anchor_vertex"), py::arg("spanning_dims"), py::arg("domain")) \
            .def_property_readonly("dim", &Cube_##DIM##D::dim) \
            .def_property_readonly("uid", &Cube_##DIM##D::get_uid, "Get UID of a cube") \
            .def_property_readonly("vertices", &Cube_##DIM##D::vertices, "Get all vertices of a cube") \
            .def_property_readonly("anchor_vertex", &Cube_##DIM##D::get_vertex, "Get anchor vertex of a cube") \
            .def_property("id", &Cube_##DIM##D::get_id, &Cube_##DIM##D::set_id, "User ID of a cube") \
            .def_property_readonly("domain", &Cube_##DIM##D::global_domain) \
            .def("boundary", &Cube_##DIM##D::boundary_cubes, "boundary of a cube") \
            .def("coboundary", &Cube_##DIM##D::coboundary_cubes, "coboundary of a cube") \
            .def("top_cofaces", &Cube_##DIM##D::top_cofaces_cubes, "top cofaces of a cube") \
            .def("__repr__", &Cube_##DIM##D::pretty_print) \
            .def("__str__", &Cube_##DIM##D::pretty_print) \
            .def(py::self == py::self) \
            .def(py::self != py::self) \
            .def(py::hash(py::self)) \


    BIND_CUBE(1);
    BIND_CUBE(2);
    BIND_CUBE(3);

    #undef BIND_CUBE
}