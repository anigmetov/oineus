#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <variant>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <oineus/timer.h>
#include <oineus/oineus.h>

namespace oin = oineus;
using dim_type = oin::dim_type;

template<class Real>
class PyOineusDiagrams {
public:
    using Coordinate = Real;

    PyOineusDiagrams() = default;

    PyOineusDiagrams(const oin::Diagrams<Real>& _diagrams)
            :diagrams_(_diagrams) { }

    PyOineusDiagrams(oin::Diagrams<Real>&& _diagrams)
            :diagrams_(_diagrams) { }

    template<class R>
    py::array_t<R> diagram_to_numpy(const typename oin::Diagrams<R>::Dgm& dgm) const
    {
        size_t arr_sz = dgm.size() * 2;
        R* ptr = new R[arr_sz];
        for(size_t i = 0 ; i < dgm.size() ; ++i) {
            ptr[2 * i] = dgm[i].birth;
            ptr[2 * i + 1] = dgm[i].death;
        }

        py::capsule free_when_done(ptr, [](void* p) {
          R* pp = reinterpret_cast<R*>(p);
          delete[] pp;
        });

        py::array::ShapeContainer shape {static_cast<long int>(dgm.size()), 2L};
        py::array::StridesContainer strides {static_cast<long int>(2 * sizeof(R)),
                                             static_cast<long int>(sizeof(R))};

        return py::array_t<R>(shape, strides, ptr, free_when_done);
    }

    py::array_t<Real> get_diagram_in_dimension_as_numpy(dim_type d)
    {
        auto dgm = diagrams_.get_diagram_in_dimension(d);
        return diagram_to_numpy<Real>(dgm);
    }

    auto get_diagram_in_dimension(dim_type d)
    {
        return diagrams_.get_diagram_in_dimension(d);
    }

    auto get_index_diagram_in_dimension(dim_type d, bool sorted = true)
    {
        return diagrams_.get_index_diagram_in_dimension(d, sorted);
    }

    py::array_t<size_t> get_index_diagram_in_dimension_as_numpy(dim_type d, bool sorted = true)
    {
        auto index_dgm = diagrams_.get_index_diagram_in_dimension(d, sorted);
        return diagram_to_numpy<size_t>(index_dgm);
    }

private:
    oin::Diagrams<Real> diagrams_;
};

template<class Int, class Real>
using DiagramV = std::pair<PyOineusDiagrams<Real>, typename oin::VRUDecomposition<Int>::MatrixData>;

template<class Int, class Real>
using DiagramRV = std::tuple<PyOineusDiagrams<Real>, typename oin::VRUDecomposition<Int>::MatrixData, typename oin::VRUDecomposition<Int>::MatrixData>;

template<class Int, class Real, size_t D>
typename oin::Grid<Int, Real, D>
get_grid(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool wrap)
{
    using Grid = oin::Grid<Int, Real, D>;
    using GridPoint = typename Grid::GridPoint;

    py::buffer_info data_buf = data.request();

    if (data.ndim() != D)
        throw std::runtime_error("Dimension mismatch");

    Real* pdata {static_cast<Real*>(data_buf.ptr)};

    GridPoint dims;
    for(dim_type d = 0 ; d < D ; ++d)
        dims[d] = data.shape(d);

    return Grid(dims, wrap, pdata);
}

template<class Int, class Real, size_t D>
decltype(auto)
get_fr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto grid = get_grid<Int, Real, D>(data, wrap);
    return grid.freudenthal_filtration(max_dim, negate, n_threads);
}

template<class Int, class Real, size_t D>
decltype(auto)
get_fr_filtration_and_critical_vertices(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto grid = get_grid<Int, Real, D>(data, wrap);
    return grid.freudenthal_filtration_and_critical_vertices(max_dim, negate, n_threads);
}

template<class Real, size_t D>
decltype(auto) numpy_to_point_vector(py::array_t<Real, py::array::c_style | py::array::forcecast> data)
{
    using PointVector = std::vector<oin::Point<Real, D>>;

    if (data.ndim() != 2 or data.shape(1) != D)
        throw std::runtime_error("Dimension mismatch");

    py::buffer_info data_buf = data.request();

    PointVector points(data.shape(0));

    Real* pdata {static_cast<Real*>(data_buf.ptr)};

    for(size_t i = 0 ; i < data.size() ; ++i)
        points[i / D][i % D] = pdata[i];

    return points;
}

template<typename Int, typename Real>
decltype(auto)
list_to_filtration(py::list data) //take a list of cells and turn it into a filtration for oineus. The list should contain cells in the form '[id, [boundary], filtration value]'.
{
    using Fil = oin::Filtration<oin::Simplex<Int>, Real>;
    using Simplex = oin::Simplex<Int>;
    using SimplexVector = typename Fil::CellVector;

    int n_simps = data.size();
    SimplexVector FSV;

    for(int i = 0 ; i < n_simps ; i++) {
        auto data_i = data[i];
        int count = 0;
        Int id;
        std::vector<Int> vertices;
        Real val;
        for(auto item: data_i) {
            if (count == 0) {
                id = item.cast<Int>();
            } else if (count == 1) {
                vertices = item.cast<std::vector<Int>>();
            } else if (count == 2) {
                val = item.cast<Real>();
            }
            count++;
        }
        FSV.emplace_back(Simplex(id, vertices), val);
    }

    return Fil(FSV, false, 1);
}

template<typename Int, typename Real>
decltype(auto)
get_ls_filtration(const py::list& simplices, const py::array_t<Real>& vertex_values, bool negate, int n_threads)
// take a list of cells and a numpy array of their values and turn it into a filtration for oineus.
// The list should contain cells, each simplex is a list of vertices,
// e.g., triangulation of one segment is [[0], [1], [0, 1]]
{
    using Fil = oin::Filtration<oin::Simplex<Int>, Real>;
    using Simplex = typename Fil::Cell;
    using IdxVector = typename Simplex::Cell::IdxVector;
    using SimplexVector = std::vector<Simplex>;

    Timer timer;
    timer.reset();

    SimplexVector fil_simplices;
    fil_simplices.reserve(simplices.size());

    if (vertex_values.ndim() != 1) {
        std::cerr << "get_ls_filtration: expected 1-dimensional array in get_ls_filtration, got " << vertex_values.ndim() << std::endl;
        throw std::runtime_error("Expected 1-dimensional array in get_ls_filtration");
    }

    auto cmp = negate ? [](Real x, Real y) { return x > y; } : [](Real x, Real y) { return x < y; };

    auto vv_buf = vertex_values.request();
    Real* p_vertex_values = static_cast<Real*>(vv_buf.ptr);

    for(auto&& item: simplices) {
        IdxVector vertices = item.cast<IdxVector>();

        Int critical_vertex;
        Real critical_value = negate ? std::numeric_limits<Real>::max() : std::numeric_limits<Real>::lowest();

        for(auto v: vertices) {
            Real vv = p_vertex_values[v];
            if (cmp(critical_value, vv)) {
                critical_vertex = v;
                critical_value = vv;
            }
        }

        fil_simplices.emplace_back(vertices, critical_value);
    }

//    std::cerr << "without filtration ctor, in get_ls_filtration elapsed: " << timer.elapsed_reset() << std::endl;

    return Fil(std::move(fil_simplices), negate, n_threads);
}

template<class Int, class Real, size_t D>
decltype(auto)
get_vr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> points, dim_type max_dim, Real max_radius, int n_threads)
{
    return oin::get_vr_filtration<Int, Real, D>(numpy_to_point_vector<Real, D>(points), max_dim, max_radius, n_threads);
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_and_critical_edges_from_pwdists(py::array_t<Real, py::array::c_style | py::array::forcecast> pw_dists, dim_type max_dim, Real max_radius, int n_threads)
{
    if (pw_dists.ndim() != 2 or pw_dists.shape(0) != pw_dists.shape(1))
        throw std::runtime_error("Dimension mismatch");

    py::buffer_info pw_dists_buf = pw_dists.request();

    Real* pdata {static_cast<Real*>(pw_dists_buf.ptr)};

    size_t n_points = pw_dists.shape(1);

    oin::DistMatrix<Real> dist_matrix {pdata, n_points};

    return oin::get_vr_filtration_and_critical_edges<Int, Real>(dist_matrix, max_dim, max_radius, n_threads);
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_from_pwdists(py::array_t<Real, py::array::c_style | py::array::forcecast> pw_dists, dim_type max_dim, Real max_radius, int n_threads)
{
    return get_vr_filtration_and_critical_edges_from_pwdists<Int, Real>(pw_dists, max_dim, max_radius, n_threads).first;
}

template<class Cell, class Real>
PyOineusDiagrams<Real>
compute_diagrams_from_fil(const oineus::Filtration<Cell, Real>& fil, int n_threads)
{
    using Int = typename Cell::Int;
    oineus::VRUDecomposition<Int> d_matrix {fil, false};

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil));
}

template<class Cell, class Real>
PyOineusDiagrams<Real>
compute_relative_diagrams(const oineus::Filtration<Cell, Real>& fil, const oineus::Filtration<Cell, Real>& relative, bool include_inf_points)
{
    using Int = typename Cell::Int;

    typename Cell::UidSet relative_;
    for(const auto& sigma : relative.cells()) {
        relative_.insert(sigma.get_uid());
    }

    auto rel_matrix = fil.boundary_matrix_full_rel(relative_);
    oineus::VRUDecomposition<Int> d_matrix {rel_matrix, false};

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = 1;

    d_matrix.reduce(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil, relative_, include_inf_points));
}

template<class Int, class Real, size_t D>
decltype(auto)
get_vr_filtration_and_critical_edges(py::array_t<Real, py::array::c_style | py::array::forcecast> points, dim_type max_dim, Real max_radius, int n_threads)
{
    return oin::get_vr_filtration_and_critical_edges<Int, Real, D>(numpy_to_point_vector<Real, D>(points), max_dim, max_radius, n_threads);
}

template<class Int, class Real, size_t D>
typename oin::VRUDecomposition<Int>::MatrixData
get_boundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads);
    return fil.boundary_matrix_full();
}

template<class Int, class Real, size_t D>
typename oin::VRUDecomposition<Int>::MatrixData
get_coboundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads);
    auto bm = fil.boundary_matrix_full();
    return oin::antitranspose(bm);
}

template<class Int, class Real, size_t D>
PyOineusDiagrams<Real>
compute_diagrams_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, oin::Params& params, bool include_inf_points, bool dualize)
{
    // for diagram in dimension d, we need (d+1)-cells
    Timer timer;
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim + 1, params.n_threads);
    auto elapsed_fil = timer.elapsed_reset();
    oin::VRUDecomposition<Int> decmp {fil, dualize};
    auto elapsed_decmp_ctor = timer.elapsed_reset();

    if (params.print_time)
        std::cerr << "Filtration: " << elapsed_fil << ", decomposition ctor: " << elapsed_decmp_ctor << std::endl;

    decmp.reduce(params);

    if (params.do_sanity_check and not decmp.sanity_check())
        throw std::runtime_error("sanity check failed");
    return PyOineusDiagrams<Real>(decmp.diagram(fil, include_inf_points));
}


template<typename C, typename Real>
decltype(auto) compute_kernel_image_cokernel_reduction(py::list K_, py::list L_, py::list IdMap_,
        oin::Params& params) //take a list of cells and turn it into a filtration for oineus. The list should contain cells in the form '[id, [boundary], filtration value].
{
    using Int = typename C::Int;
    using Filtration = oin::Filtration<C, Real>;
    using KICR = oin::KerImCokReduced<C, Real, 2>;

    std::cout << "======================================" << std::endl;
    std::cout << std::endl;
    std::cout << "You have called \'compute_kernel_image_cokernel_reduction\', it takes as input a complex K, and a subcomplex L, as lists of cells in the format:" << std::endl;
    std::cout << "          [id, [boundary], filtration value]" << std::endl;
    std::cout << "and a mapping from L to K, which takes the id of a cell in L and returns the id of the cell in K, as well as an integer, telling oineus how many threads to use." << std::endl;
    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    std::cout << "------------ Importing K ------------" << std::endl;
    Filtration K = list_to_filtration<Int, Real>(K_);
    std::cout << "------------ Importing L ------------" << std::endl;
    Filtration L = list_to_filtration<Int, Real>(L_);

    int n_L = IdMap_.size();
    std::vector<int> IdMapping;

    for(int i = 0 ; i < n_L ; i++) {
        int i_map = IdMap_[i].cast<int>();
        IdMapping.push_back(i_map);
    }

    if (params.verbose) {
        std::cout << "---------- Map from L to K ----------" << std::endl;
        for(int i = 0 ; i < n_L ; i++) {
            std::cout << "Cell " << i << " in L is mapped to cell " << IdMapping[i] << " in K." << std::endl;
        }
    }

    params.sort_dgms = false;
    params.clearing_opt = false;

    oin::KICRParams kicr_params;
    kicr_params.verbose = params.verbose;
    kicr_params.kernel = kicr_params.image = kicr_params.cokernel = true;
    kicr_params.params_f = kicr_params.params_g = params;
    kicr_params.params_ker = kicr_params.params_cok = kicr_params.params_im = params;

    KICR result { K, L, kicr_params };

    return result;
}


template<class Int>
void init_oineus_common(py::module& m)
{
    using namespace pybind11::literals;

    using oin::VREdge;

    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;
    using Decomposition = oin::VRUDecomposition<Int>;
    using ReductionParams = oin::Params;
    using KICRParams = oin::KICRParams;
    std::string vr_edge_name = "VREdge";

    py::class_<VREdge>(m, vr_edge_name.c_str())
            .def(py::init<Int>())
            .def_readwrite("x", &VREdge::x)
            .def_readwrite("y", &VREdge::y)
            .def("__getitem__", [](const VREdge& p, int i) {
              if (i == 0)
                  return p.x;
              else if (i == 1)
                  return p.y;
              else
                  throw std::out_of_range("i must be 0 or 1");
            })
            .def("__repr__", [](const VREdge& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<ReductionParams>(m, "ReductionParams")
            .def(py::init<>())
            .def_readwrite("n_threads", &ReductionParams::n_threads)
            .def_readwrite("chunk_size", &ReductionParams::chunk_size)
            .def_readwrite("write_dgms", &ReductionParams::write_dgms)
            .def_readwrite("sort_dgms", &ReductionParams::sort_dgms)
            .def_readwrite("clearing_opt", &ReductionParams::clearing_opt)
            .def_readwrite("acq_rel", &ReductionParams::acq_rel)
            .def_readwrite("print_time", &ReductionParams::print_time)
            .def_readwrite("elapsed", &ReductionParams::elapsed)
            .def_readwrite("compute_v", &ReductionParams::compute_v)
            .def_readwrite("compute_u", &ReductionParams::compute_u)
            .def_readwrite("do_sanity_check", &ReductionParams::do_sanity_check)
            .def_readwrite("verbose", &ReductionParams::verbose)
            .def("__repr__", [](const ReductionParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(py::pickle(
                    // __getstate__

                    [](const ReductionParams& p) {
                      return py::make_tuple(p.n_threads, p.chunk_size, p.write_dgms,
                              p.sort_dgms, p.clearing_opt, p.acq_rel, p.print_time, p.compute_v, p.compute_u,
                              p.do_sanity_check, p.elapsed, p.verbose);
                    },
                    // __setstate__
                    [](py::tuple t) {
                      if (t.size() != 12)
                          throw std::runtime_error("Invalid tuple for ReductionParams");

                      ReductionParams p;

                      int i = 0;

                      p.n_threads = t[i++].cast<decltype(p.n_threads)>();
                      p.chunk_size = t[i++].cast<decltype(p.chunk_size)>();
                      p.write_dgms = t[i++].cast<decltype(p.write_dgms)>();
                      p.sort_dgms = t[i++].cast<decltype(p.sort_dgms)>();
                      p.clearing_opt = t[i++].cast<decltype(p.clearing_opt)>();
                      p.acq_rel = t[i++].cast<decltype(p.acq_rel)>();
                      p.print_time = t[i++].cast<decltype(p.print_time)>();
                      p.compute_v = t[i++].cast<decltype(p.compute_v)>();
                      p.compute_u = t[i++].cast<decltype(p.compute_u)>();
                      p.do_sanity_check = t[i++].cast<decltype(p.do_sanity_check)>();

                      p.elapsed = t[i++].cast<decltype(p.elapsed)>();
                      p.verbose = t[i++].cast<decltype(p.verbose)>();

                      return p;
                    }));

    py::class_<KICRParams>(m, "KICRParams")
            .def(py::init<>())
            .def_readwrite("kernel", &KICRParams::kernel)
            .def_readwrite("image", &KICRParams::image)
            .def_readwrite("cokernel", &KICRParams::cokernel)
            .def_readwrite("include_zero_persistence", &KICRParams::include_zero_persistence)
            .def_readwrite("verbose", &KICRParams::verbose)
            .def_readwrite("params_f", &KICRParams::params_f)
            .def_readwrite("params_g", &KICRParams::params_g)
            .def_readwrite("params_ker", &KICRParams::params_ker)
            .def_readwrite("params_im", &KICRParams::params_ker)
            .def_readwrite("params_cok", &KICRParams::params_ker)
            .def("__repr__", [](const KICRParams& self) { std::stringstream ss; ss << self; return ss.str(); })
            .def(py::pickle(
                    // __getstate__

                    [](const KICRParams& p) {
                      return py::make_tuple(p.kernel, p.image, p.cokernel, p.include_zero_persistence, p.verbose,
                              p.params_f, p.params_g, p.params_ker, p.params_im, p.params_cok);
                    },
                    // __setstate__
                    [](py::tuple t) {
                      if (t.size() != 10)
                          throw std::runtime_error("Invalid tuple for KICRParams");

                      KICRParams p;

                      int i = 0;

                      p.kernel = t[i++].cast<decltype(p.kernel)>();
                      p.image = t[i++].cast<decltype(p.image)>();
                      p.cokernel = t[i++].cast<decltype(p.cokernel)>();
                      p.include_zero_persistence = t[i++].cast<decltype(p.include_zero_persistence)>();
                      p.verbose = t[i++].cast<decltype(p.verbose)>();
                      p.params_f = t[i++].cast<decltype(p.params_f)>();
                      p.params_g = t[i++].cast<decltype(p.params_g)>();
                      p.params_ker = t[i++].cast<decltype(p.params_ker)>();
                      p.params_im = t[i++].cast<decltype(p.params_im)>();
                      p.params_cok = t[i++].cast<decltype(p.params_cok)>();

                      return p;
                    }));

    py::enum_<DenoiseStrategy>(m, "DenoiseStrategy", py::arithmetic())
            .value("BirthBirth", DenoiseStrategy::BirthBirth, "(b, d) maps to (b, b)")
            .value("DeathDeath", DenoiseStrategy::DeathDeath, "(b, d) maps to (d, d)")
            .value("Midway", DenoiseStrategy::Midway, "((b, d) maps to ((b+d)/2, (b+d)/2)")
            .def("as_str", [](const DenoiseStrategy& self) { return denoise_strategy_to_string(self); });

    py::enum_<ConflictStrategy>(m, "ConflictStrategy", py::arithmetic())
            .value("Max", ConflictStrategy::Max, "choose maximal displacement")
            .value("Avg", ConflictStrategy::Avg, "average gradients")
            .value("Sum", ConflictStrategy::Sum, "sum gradients")
            .value("FixCritAvg", ConflictStrategy::FixCritAvg, "use matching on critical, average gradients on other cells")
            .def("as_str", [](const ConflictStrategy& self) { return conflict_strategy_to_string(self); });

    using Simp = oin::Simplex<Int>;

    py::class_<Simp>(m, "Simplex")
            .def(py::init([](typename Simp::IdxVector vs) -> Simp {
                      return Simp({vs});
                    }),
                    py::arg("vertices"))
            .def(py::init([](Int id, typename Simp::IdxVector vs) -> Simp {
              return Simp(id, vs);
            }), py::arg("id"), py::arg("vertices"))
            .def_property("id", &Simp::get_id, &Simp::set_id)
            .def_property("vertices", &Simp::get_uid, &Simp::set_uid)
            .def("get_uid", &Simp::get_uid)
            .def("dim", &Simp::dim)
            .def("boundary", &Simp::boundary)
            .def("join", [](const Simp& sigma, Int new_vertex, Int new_id) {
                      return sigma.join(new_id, new_vertex);
                    },
                    py::arg("new_vertex"),
                    py::arg("new_id") = Simp::k_invalid_id)
            .def("__repr__", [](const Simp& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    using ProdSimplex = oin::ProductCell<Simp, Simp>;

    py::class_<ProdSimplex>(m, "SimplexProduct")
            .def(py::init([](const Simp& s1, const Simp& s2) -> ProdSimplex {
                      return {s1, s2};
                    }),
                    py::arg("simplex_1"), py::arg("simplex_2"))
            .def_property("id", &ProdSimplex::get_id, &ProdSimplex::set_id)
            .def("get_uid", &ProdSimplex::get_uid)
            .def("dim", &ProdSimplex::dim)
            .def("boundary", &ProdSimplex::boundary)
            .def("__repr__", [](const ProdSimplex& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

}

void init_oineus_common_int(py::module& m);

template<class Int>
void init_oineus_common_decomposition(py::module& m)
{
    using namespace pybind11::literals;

    using oin::VREdge;
    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;
    using Decomposition = oin::VRUDecomposition<Int>;
    using Simplex = oin::Simplex<Int>;
    using SimplexFiltrationDouble = oin::Filtration<Simplex, double>;
    using SimplexFiltrationFloat = oin::Filtration<Simplex, float>;
    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexFiltrationDouble = oin::Filtration<ProdSimplex, double>;
    using ProdSimplexFiltrationFloat = oin::Filtration<ProdSimplex, float>;
    std::string vr_edge_name = "VREdge";

    py::class_<Decomposition>(m, "Decomposition")
            .def(py::init<const SimplexFiltrationDouble&, bool>())
            .def(py::init<const SimplexFiltrationFloat&, bool>())
            .def(py::init<const ProdSimplexFiltrationDouble&, bool>())
            .def(py::init<const ProdSimplexFiltrationFloat&, bool>())
            .def(py::init<const typename Decomposition::MatrixData&, size_t, bool>())
            .def_readwrite("r_data", &Decomposition::r_data)
            .def_readwrite("v_data", &Decomposition::v_data)
            .def_readwrite("u_data_t", &Decomposition::u_data_t)
            .def_readwrite("d_data", &Decomposition::d_data)
            .def("reduce", &Decomposition::reduce, py::call_guard<py::gil_scoped_release>())
            .def("sanity_check", &Decomposition::sanity_check, py::call_guard<py::gil_scoped_release>())
            .def("diagram", [](const Decomposition& self, const SimplexFiltrationDouble& fil, bool include_inf_points) { return PyOineusDiagrams<double>(self.diagram(fil, include_inf_points)); },
                    py::arg("fil"), py::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const SimplexFiltrationFloat& fil, bool include_inf_points) { return PyOineusDiagrams<float>(self.diagram(fil, include_inf_points)); },
                    py::arg("fil"), py::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const ProdSimplexFiltrationDouble& fil, bool include_inf_points) { return PyOineusDiagrams<double>(self.diagram(fil, include_inf_points)); },
                    py::arg("fil"), py::arg("include_inf_points") = true)
            .def("diagram", [](const Decomposition& self, const ProdSimplexFiltrationFloat& fil, bool include_inf_points) { return PyOineusDiagrams<float>(self.diagram(fil, include_inf_points)); },
                    py::arg("fil"), py::arg("include_inf_points") = true)
            .def("zero_pers_diagram", [](const Decomposition& self, const SimplexFiltrationFloat& fil) { return PyOineusDiagrams<float>(self.zero_persistence_diagram(fil)); },
                    py::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const SimplexFiltrationDouble& fil) { return PyOineusDiagrams<double>(self.zero_persistence_diagram(fil)); },
                    py::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const ProdSimplexFiltrationFloat& fil) { return PyOineusDiagrams<float>(self.zero_persistence_diagram(fil)); },
                    py::arg("fil"))
            .def("zero_pers_diagram", [](const Decomposition& self, const ProdSimplexFiltrationDouble& fil) { return PyOineusDiagrams<double>(self.zero_persistence_diagram(fil)); },
                    py::arg("fil"))
            .def("filtration_index", &Decomposition::filtration_index, py::arg("matrix_index"))
                    ;

}

void init_oineus_common_decomposition_int(py::module& m);

template<class Int>
void init_oineus_common_diagram(py::module& m)
{
    using namespace pybind11::literals;

    using oin::VREdge;
    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;

    using DgmPointInt = typename oin::DgmPoint<Int>;
    using DgmPointSizet = typename oin::DgmPoint<size_t>;

    py::class_<DgmPointInt>(m, "DgmPoint_int")
            .def(py::init<Int, Int>())
            .def_readwrite("birth", &DgmPointInt::birth)
            .def_readwrite("death", &DgmPointInt::death)
            .def("__getitem__", [](const DgmPointInt& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPointInt& p) { return std::hash<DgmPointInt>()(p); })
            .def("__eq__", [](const DgmPointInt& p, const DgmPointInt& q) { return p == q; })
            .def("__repr__", [](const DgmPointInt& p) {
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

void init_oineus_common_diagram_int(py::module& m);

template<class Int, class Real>
void init_oineus_functions(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using Simp = oin::Simplex<Int>;
    using SimpProd = oin::ProductCell<Simp, Simp>;
    using Filtration = oin::Filtration<Simp, Real>;

    using oin::VREdge;

    std::string func_name;

    // diagrams
    func_name = "compute_diagrams_ls" + suffix + "_1";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<Int, Real, 1>);

    func_name = "compute_diagrams_ls" + suffix + "_2";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<Int, Real, 2>);

    func_name = "compute_diagrams_ls" + suffix + "_3";
    m.def(func_name.c_str(), &compute_diagrams_ls_freudenthal<Int, Real, 3>);

    // Lower-star Freudenthal filtration
    func_name = "get_fr_filtration" + suffix + "_1";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 1>);

    func_name = "get_fr_filtration" + suffix + "_2";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 2>);

    func_name = "get_fr_filtration" + suffix + "_3";
    m.def(func_name.c_str(), &get_fr_filtration<Int, Real, 3>);

    func_name = "get_fr_filtration_and_critical_vertices" + suffix + "_1";
    m.def(func_name.c_str(), &get_fr_filtration_and_critical_vertices<Int, Real, 1>);

    func_name = "get_fr_filtration_and_critical_vertices" + suffix + "_2";
    m.def(func_name.c_str(), &get_fr_filtration_and_critical_vertices<Int, Real, 2>);

    func_name = "get_fr_filtration_and_critical_vertices" + suffix + "_3";
    m.def(func_name.c_str(), &get_fr_filtration_and_critical_vertices<Int, Real, 3>);

    // Vietoris--Rips filtration
    func_name = "get_vr_filtration" + suffix + "_1";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 1>);

    func_name = "get_vr_filtration" + suffix + "_2";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 2>);

    func_name = "get_vr_filtration" + suffix + "_3";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 3>);

    func_name = "get_vr_filtration" + suffix + "_4";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 4>);

    func_name = "get_vr_filtration_and_critical_edges" + suffix + "_1";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges<Int, Real, 1>);

    func_name = "get_vr_filtration_and_critical_edges" + suffix + "_2";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges<Int, Real, 2>);

    func_name = "get_vr_filtration_and_critical_edges" + suffix + "_3";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges<Int, Real, 3>);

    func_name = "get_vr_filtration_and_critical_edges" + suffix + "_4";
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges<Int, Real, 4>);

    func_name = "get_vr_filtration_from_pwdists" + suffix;
    m.def(func_name.c_str(), &get_vr_filtration_from_pwdists<Int, Real>);

    func_name = "get_vr_filtration_and_critical_edges_from_pwdists" + suffix;
    m.def(func_name.c_str(), &get_vr_filtration_and_critical_edges_from_pwdists<Int, Real>);


    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix" + suffix + "_1";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 1>);

    func_name = "get_boundary_matrix" + suffix + "_2";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 2>);

    func_name = "get_boundary_matrix" + suffix + "_3";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 3>);

    // target values
    func_name = "get_denoise_target" + suffix;
    m.def(func_name.c_str(), &oin::get_denoise_target<Simp, Real>);

    // target values -- diagram loss
    func_name = "get_target_values_diagram_loss" + suffix;
    m.def(func_name.c_str(), &oin::get_prescribed_simplex_values_diagram_loss<Real>);

    // target values --- X set
    func_name = "get_target_values_x" + suffix;
    m.def(func_name.c_str(), &oin::get_prescribed_simplex_values_set_x<Simp, Real>);

    // to reproduce "Well group loss" experiments
    func_name = "get_well_group_target" + suffix;
    m.def(func_name.c_str(), &oin::get_well_group_target<Simp, Real>);

    func_name = "get_nth_persistence" + suffix;
    m.def(func_name.c_str(), &oin::get_nth_persistence<Simp, Real>);

    // to get permutation for Warm Starts
    func_name = "get_permutation" + suffix;
    m.def(func_name.c_str(), &oin::targets_to_permutation<Simp, Real>);

    func_name = "get_permutation_dtv" + suffix;
    m.def(func_name.c_str(), &oin::targets_to_permutation_dtv<Simp, Real>);

    func_name = "list_to_filtration" + suffix;
    m.def(func_name.c_str(), &list_to_filtration<Int, Real>);

    func_name = "compute_kernel_image_cokernel_reduction" + suffix;
    m.def(func_name.c_str(), &compute_kernel_image_cokernel_reduction<Simp, Real>);

    func_name = "get_ls_filtration" + suffix;
    m.def(func_name.c_str(), &get_ls_filtration<Int, Real>);

    func_name = "compute_relative_diagrams" + suffix;
    m.def(func_name.c_str(), &compute_relative_diagrams<Simp, Real>, py::arg("fil"), py::arg("rel"), py::arg("include_inf_points")=true);

    func_name = "compute_relative_diagrams" + suffix;
    m.def(func_name.c_str(), &compute_relative_diagrams<SimpProd, Real>, py::arg("fil"), py::arg("rel"), py::arg("include_inf_points")=true);
}

void init_oineus_functions_double(py::module& m, std::string suffix);
void init_oineus_functions_float(py::module& m, std::string suffix);

template<class Int, class Real>
void init_oineus_fil_dgm_simplex(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using DgmPoint = typename oin::Diagrams<Real>::Point;
    using DgmPtVec = typename oin::Diagrams<Real>::Dgm;
    using IndexDgmPtVec = typename oin::Diagrams<Real>::IndexDgm;
    using Diagram = PyOineusDiagrams<Real>;

    using Simplex = oin::Simplex<Int>;
    using SimplexValue = oin::CellWithValue<oin::Simplex<Int>, Real>;
    using Filtration = oin::Filtration<Simplex, Real>;

    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdSimplexValue = oin::CellWithValue<ProdSimplex, Real>;
    using ProdFiltration = oin::Filtration<ProdSimplex, Real>;

    using oin::VREdge;

    using VRUDecomp = oin::VRUDecomposition<Int>;
    using KerImCokRedSimplex = oin::KerImCokReduced<Simplex, Real>;
    using KerImCokRedProdSimplex = oin::KerImCokReduced<ProdSimplex, Real>;

    std::string filtration_class_name = "Filtration" + suffix;
    std::string prod_filtration_class_name = "ProdFiltration" + suffix;
    std::string simplex_class_name = "Simplex" + suffix;
    std::string prod_simplex_class_name = "ProdSimplex" + suffix;

    std::string dgm_point_name = "DiagramPoint" + suffix;
    std::string dgm_class_name = "Diagrams" + suffix;

    std::string ker_im_cok_reduced_class_name = "KerImCokReduced" + suffix;
    std::string ker_im_cok_reduced_prod_class_name = "KerImCokReducedProd" + suffix;

    py::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<Real, Real>(), py::arg("birth"), py::arg("death"))
            .def_readwrite("birth", &DgmPoint::birth)
            .def_readwrite("death", &DgmPoint::death)
            .def_readwrite("birth_index", &DgmPoint::birth_index)
            .def_readwrite("death_index", &DgmPoint::death_index)
            .def("__getitem__", [](const DgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPoint& p) { return std::hash<DgmPoint>()(p); })
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<Diagram>(m, dgm_class_name.c_str())
            .def(py::init<dim_type>())
            .def("in_dimension", [](Diagram& self, dim_type dim, bool as_numpy) -> std::variant<pybind11::array_t<Real>, DgmPtVec> {
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

    py::class_<SimplexValue>(m, simplex_class_name.c_str())
            .def(py::init([](typename Simplex::IdxVector vs, Real value) -> SimplexValue {
                      return SimplexValue({vs}, value);
                    }),
                    py::arg("vertices"),
                    py::arg("value"))
            .def(py::init([](Int id, typename Simplex::IdxVector vs, Real value) -> SimplexValue { return SimplexValue({id, vs}, value); }), py::arg("id"), py::arg("vertices"), py::arg("value"))
            .def_property("id", &SimplexValue::get_id, &SimplexValue::set_id)
            .def_readwrite("sorted_id", &SimplexValue::sorted_id_)
            .def_property("vertices", &SimplexValue::get_uid, &SimplexValue::set_uid)
            .def_readwrite("value", &SimplexValue::value_)
            .def("dim", &SimplexValue::dim)
            .def("boundary", &SimplexValue::boundary)
            .def("join", [](const SimplexValue& sigma, Int new_vertex, Real value, Int new_id) {
                      return sigma.join(new_id, new_vertex, value);
                    },
                    py::arg("new_vertex"),
                    py::arg("value"),
                    py::arg("new_id") = SimplexValue::k_invalid_id)
            .def("__repr__", [](const SimplexValue& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    py::class_<ProdSimplexValue>(m, prod_simplex_class_name.c_str())
            .def(py::init([](const SimplexValue& sigma, const SimplexValue& tau, Real value) -> ProdSimplexValue {
                      return ProdSimplexValue(ProdSimplex(sigma.get_cell(), tau.get_cell()), value);
                    }),
                    py::arg("cell_1"),
                    py::arg("cell_2"),
                    py::arg("value"))
            .def(py::init([](const Simplex& sigma, const Simplex& tau, Real value) -> ProdSimplexValue {
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
            .def("__repr__", [](const ProdSimplexValue& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    py::class_<Filtration>(m, filtration_class_name.c_str())
            .def(py::init<typename Filtration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate") = false,
                    py::arg("n_threads") = 1,
                    py::arg("sort_only_by_dimension") = false,
                    py::arg("set_ids") = true)
            .def("max_dim", &Filtration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &Filtration::cells_copy, "copy of all cells in filtration order")
            .def("simplices", &Filtration::cells_copy, "copy of all simplices (cells) in filtration order")
            .def("size", &Filtration::size, "number of cells in filtration")
            .def("__len__", &Filtration::size)
            .def("size_in_dimension", &Filtration::size_in_dimension, py::arg("dim"), "number of cells of dimension dim")
            .def("n_vertices", &Filtration::n_vertices)
            .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, py::arg("sorted_id"))
            .def("get_id_by_sorted_id", &Filtration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("get_sorted_id_by_id", &Filtration::get_sorted_id, py::arg("id"))
            .def("get_cell", &Filtration::get_cell, py::arg("i"))
            .def("get_simplex", &Filtration::get_cell, py::arg("i"))
            .def("get_sorting_permutation", &Filtration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .def("simplex_value_by_vertices", &Filtration::value_by_vertices, py::arg("vertices"))
            .def("get_sorted_id_by_vertices", &Filtration::get_sorted_id_by_vertices, py::arg("vertices"))
            .def("cell_by_uid", &Filtration::get_cell_by_uid, py::arg("uid"))
            .def("boundary_matrix", &Filtration::boundary_matrix_full)
            .def("coboundary_matrix", &Filtration::coboundary_matrix)
            .def("boundary_matrix_rel", &Filtration::boundary_matrix_full_rel)
            .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            .def("set_values", &Filtration::set_values)
            .def("__iter__", [](Filtration& fil) { return py::make_iterator(fil.begin(), fil.end()); }, py::keep_alive<0, 1>())
            .def("subfiltration", [](Filtration& self, const py::function& py_pred) {
                auto pred = [&py_pred](const typename Filtration::Cell& s) -> bool { return py_pred(s).template cast<bool>(); };
                Filtration result = self.subfiltration(pred);
                return result;
            }, py::arg("predicate"), py::return_value_policy::move)
            .def("__repr__", [](const Filtration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            });

    py::class_<ProdFiltration>(m, prod_filtration_class_name.c_str())
            .def(py::init<typename ProdFiltration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate") = false,
                    py::arg("n_threads") = 1,
                    py::arg("sort_only_by_dimension") = false,
                    py::arg("set_ids") = true)
            .def("max_dim", &ProdFiltration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &ProdFiltration::cells_copy, "copy of all cells in filtration order")
            .def("size", &ProdFiltration::size, "number of cells in filtration")
            .def("__len__", &ProdFiltration::size)
            .def("__iter__", [](ProdFiltration& fil) { return py::make_iterator(fil.begin(), fil.end()); }, py::keep_alive<0, 1>())
            .def("size_in_dimension", &ProdFiltration::size_in_dimension, py::arg("dim"), "number of cells of dimension dim")
            .def("cell_value_by_sorted_id", &ProdFiltration::value_by_sorted_id, py::arg("sorted_id"))
            .def("get_id_by_sorted_id", &ProdFiltration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("get_sorted_id_by_id", &ProdFiltration::get_sorted_id, py::arg("id"))
            .def("get_cell", &ProdFiltration::get_cell, py::arg("i"))
            .def("get_sorting_permutation", &ProdFiltration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &ProdFiltration::get_inv_sorting_permutation)
            .def("boundary_matrix", &ProdFiltration::boundary_matrix_full)
            .def("reset_ids_to_sorted_ids", &ProdFiltration::reset_ids_to_sorted_ids)
            .def("set_values", &ProdFiltration::set_values)
            .def("__repr__", [](const ProdFiltration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            });

    m.def("mapping_cylinder", &oin::build_mapping_cylinder<Simplex, Real>, py::arg("fil_domain"), py::arg("fil_codomain"), py::arg("v_domain"), py::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("mapping_cylinder_with_indices", &oin::build_mapping_cylinder_with_indices<Simplex, Real>, py::arg("fil_domain"), py::arg("fil_codomain"), py::arg("v_domain"), py::arg("v_codomain"), "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain, fil_domain is multiplied by v_domain and fil_codomain is multiplied by v_codomain");
    m.def("multiply_filtration", &oin::multiply_filtration<Simplex, Real>, py::arg("fil"), py::arg("sigma"), "return a filtration with each simplex in fil multiplied by simplex sigma");
//    m.def("kernel_diagrams", &compute_kernel_diagrams<ProdSimplex, Real>, py::arg("fil_L"), py::arg("fil_K"), "return kernel persistence diagrams of inclusion fil_L -> fil_K");
//    m.def("kernel_cokernel_diagrams", &compute_kernel_cokernel_diagrams<ProdSimplex, Real>, py::arg("fil_L"), py::arg("fil_K"), "return kernel and cokernel persistence diagrams of inclusion fil_L -> fil_K");

    m.def("min_filtration", &oin::min_filtration<Simplex, Real>, py::arg("fil_1"), py::arg("fil_2"), "return a filtration where each simplex has minimal value from fil_1, fil_2");
    m.def("min_filtration", &oin::min_filtration<ProdSimplex, Real>, py::arg("fil_1"), py::arg("fil_2"), "return a filtration where each cell has minimal value from fil_1, fil_2");

    // helper for differentiable filtration
    m.def("min_filtration_with_indices", &oin::min_filtration_with_indices<Simplex, Real>, py::arg("fil_1"), py::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
    m.def("min_filtration_with_indices", &oin::min_filtration_with_indices<ProdSimplex, Real>, py::arg("fil_1"), py::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");

    py::class_<KerImCokRedSimplex>(m, ker_im_cok_reduced_class_name.c_str())
            .def(py::init<const Filtration&, const Filtration&, oin::KICRParams&>(), py::arg("K"), py::arg("L"), py::arg("params"))
            .def("domain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<Real>(self.get_domain_diagrams()); })
            .def("codomain_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<Real>(self.get_codomain_diagrams()); })
            .def("kernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<Real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<Real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedSimplex& self) { return PyOineusDiagrams<Real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_readwrite("decomposition_f", &KerImCokRedSimplex::dcmp_F_)
            .def_readwrite("decomposition_g", &KerImCokRedSimplex::dcmp_G_)
            .def_readwrite("decomposition_im", &KerImCokRedSimplex::dcmp_im_)
            .def_readwrite("decomposition_ker", &KerImCokRedSimplex::dcmp_ker_)
            .def_readwrite("decomposition_cok", &KerImCokRedSimplex::dcmp_cok_)
            ;

     py::class_<KerImCokRedProdSimplex>(m, ker_im_cok_reduced_prod_class_name.c_str())
            .def(py::init<const ProdFiltration&, const ProdFiltration&, oin::KICRParams&>(), py::arg("K"), py::arg("L"), py::arg("params"))
            .def("kernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<Real>(self.get_kernel_diagrams()); })
            .def("cokernel_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<Real>(self.get_cokernel_diagrams()); })
            .def("image_diagrams", [](const KerImCokRedProdSimplex& self) { return PyOineusDiagrams<Real>(self.get_image_diagrams()); })
            // decomposition objects provide access to their R/V/U matrices
            .def_readwrite("decomposition_f", &KerImCokRedProdSimplex::dcmp_F_)
            .def_readwrite("decomposition_g", &KerImCokRedProdSimplex::dcmp_G_)
            .def_readwrite("decomposition_im", &KerImCokRedProdSimplex::dcmp_im_)
            .def_readwrite("decomposition_ker", &KerImCokRedProdSimplex::dcmp_ker_)
            .def_readwrite("decomposition_cok", &KerImCokRedProdSimplex::dcmp_cok_)
            ;
}

void init_oineus_fil_dgm_simplex_float(py::module& m, std::string suffix);
void init_oineus_fil_dgm_simplex_double(py::module& m, std::string suffix);

void init_oineus_top_optimizer(py::module& m);

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
