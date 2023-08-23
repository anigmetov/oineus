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

    py::array_t<Real> diagram_to_numpy(const typename oin::Diagrams<Real>::Dgm& dgm) const
    {
        size_t arr_sz = dgm.size() * 2;
        Real* ptr = new Real[arr_sz];
        for(size_t i = 0 ; i < dgm.size() ; ++i) {
            ptr[2 * i] = dgm[i].birth;
            ptr[2 * i + 1] = dgm[i].death;
        }

        py::capsule free_when_done(ptr, [](void* p) {
          Real* pp = reinterpret_cast<Real*>(p);
          delete[] pp;
        });

        py::array::ShapeContainer shape {static_cast<long int>(dgm.size()), 2L};
        py::array::StridesContainer strides {static_cast<long int>(2 * sizeof(Real)),
                                             static_cast<long int>(sizeof(Real))};

        return py::array_t<Real>(shape, strides, ptr, free_when_done);
    }

    py::array_t<Real> get_diagram_in_dimension_as_numpy(dim_type d) const
    {
        auto dgm = diagrams_.get_diagram_in_dimension(d);
        return diagram_to_numpy(dgm);
    }

    auto get_diagram_in_dimension(dim_type d) const
    {
        return diagrams_.get_diagram_in_dimension(d);
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
list_to_filtration(py::list data, oin::Params& params) //take a list of cells and turn it into a filtration for oineus. The list should contain cells in the form '[id, [boundary], filtration value]'.
{
    using Simplex = oin::Simplex<Int, Real>;
    using Fil = oin::Filtration<Simplex>;
    using SimplexVector = typename Fil::CellVector;

    int n_simps = data.size();
    std::cout << "Number of cells in the complex is: " << n_simps << std::endl;
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
        if (params.verbose) std::cout << "parsed the following data. id: " << id << " val: " << val << std::endl;
        Simplex simp_i(id, vertices, val);
        FSV.push_back(simp_i);
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
    using Simplex = oin::Simplex<Int, Real>;
    using Fil = oin::Filtration<Simplex>;
    using IdxVector = std::vector<Int>;
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

    oin::DistMatrix<Real> dist_matrix { pdata, n_points };

    return oin::get_vr_filtration_and_critical_edges<Int, Real>(dist_matrix, max_dim, max_radius, n_threads);
}

template<class Int, class Real>
decltype(auto) get_vr_filtration_from_pwdists(py::array_t<Real, py::array::c_style | py::array::forcecast> pw_dists, dim_type max_dim, Real max_radius, int n_threads)
{
    return get_vr_filtration_and_critical_edges_from_pwdists<Int, Real>(pw_dists, max_dim, max_radius, n_threads).first;
}



template<class Int, class Real, class L>
PyOineusDiagrams<Real>
compute_diagrams_from_fil(const oineus::Filtration<oineus::Simplex<Int, Real>>& fil, int n_threads)
{
    oineus::VRUDecomposition<Int> d_matrix {fil, false};

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    return PyOineusDiagrams<Real>(d_matrix.diagram(fil));
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

template<typename Int, typename Real>
class PyKerImCokDgms {
private:
    oin::KerImCokReduced<Int, Real> KICR;
public:

    PyKerImCokDgms(oin::KerImCokReduced<Int, Real> KICR_)
            :
            KICR(KICR_)
    {

    }

    decltype(auto) kernel()
    {
        return PyOineusDiagrams<Real>(KICR.get_kernel_diagrams());
    }

    decltype(auto) image()
    {
        return PyOineusDiagrams(KICR.get_image_diagrams());
    }

    decltype(auto) cokernel()
    {
        return PyOineusDiagrams(KICR.get_cokernel_diagrams());
    }

};

template<typename Int, typename Real>
class PyKerImCokRed {
private:
    oin::KerImCokReduced<Int, Real> KICR;

public:
    bool kernel {false};
    bool image {false};
    bool cokernel {false};

    PyKerImCokRed(oin::KerImCokReduced<Int, Real> KICR_)
            :
            KICR(KICR_)
    {
    }

    decltype(auto) kernel_diagrams()
    {
        if (!kernel) {
            KICR.GenerateKerDiagrams();
            kernel = true;
        }
        return PyOineusDiagrams<Real>(KICR.get_kernel_diagrams());
    }

    decltype(auto) image_diagrams()
    {
        if (!image) {
            KICR.GenerateImDiagrams();
            image = true;
        }
        return PyOineusDiagrams(KICR.get_image_diagrams());
    }

    decltype(auto) cokernel_diagrams()
    {
        if (!cokernel) {
            KICR.GenerateCokDiagrams();
            cokernel = true;
        }
        return PyOineusDiagrams(KICR.get_cokernel_diagrams());
    }


	decltype(auto) D_F() {
		return py::cast(KICR.get_D_f());
	}

	decltype(auto) D_G() {
		return py::cast(KICR.get_D_g());
	}

	decltype(auto) D_Ker() {
		return py::cast(KICR.get_D_ker());
	}

	decltype(auto) D_Im() {
		return py::cast(KICR.get_D_im());
	}

	decltype(auto) D_Cok() {
		return py::cast(KICR.get_D_cok());
	}

	decltype(auto) R_F() {
		return py::cast(KICR.get_R_f());
	}

	decltype(auto) R_G() {
		return py::cast(KICR.get_R_g());
	}

	decltype(auto) R_Ker() {
		return py::cast(KICR.get_R_ker());
	}

	decltype(auto) R_Im() {
		return py::cast(KICR.get_R_im());
	}

	decltype(auto) R_Cok() {
		return py::cast(KICR.get_V_cok());
	}

	decltype(auto) V_F() {
		return py::cast(KICR.get_V_f());
	}

	decltype(auto) V_G() {
		return py::cast(KICR.get_V_g());
	}

	decltype(auto) V_Ker() {
		return py::cast(KICR.get_V_ker());
	}

	decltype(auto) V_Im() {
		return py::cast(KICR.get_V_im());
	}

	decltype(auto) V_Cok() {
		return py::cast(KICR.get_V_cok());
	}
};

template<typename Int, typename Real>
decltype(auto) compute_kernel_image_cokernel_reduction(py::list K_, py::list L_, py::list IdMap_,
        oin::Params& params) //take a list of cells and turn it into a filtration for oineus. The list should contain cells in the form '[id, [boundary], filtration value].
{
    using IdxVector = std::vector<Int>;
    using IntSparseColumn = oin::SparseColumn<Int>;
    using MatrixData = std::vector<IntSparseColumn>;
    using FiltrationSimplex = oin::Simplex<Int, Real>;
    using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
    using Filtration = oin::Filtration<FiltrationSimplex>;
    using KerImCokReduced = oin::KerImCokReduced<Int, Real>;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;
    std::cout << "You have called \'compute_kernel_image_cokernel_reduction\', it takes as input a complex K, and a subcomplex L, as lists of cells in the format:" << std::endl;
    std::cout << "          [id, [boundary], filtration value]" << std::endl;
    std::cout << "and a mapping from L to K, which takes the id of a cell in L and returns the id of the cell in K, as well as an integer, telling oineus how many threads to use." << std::endl;
    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    std::cout << "------------ Importing K ------------" << std::endl;
    Filtration K = list_to_filtration<Int, Real>(K_, params);
    std::cout << "------------ Importing L ------------" << std::endl;
    Filtration L = list_to_filtration<Int, Real>(L_, params);

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

    PyKerImCokRed KICR(oin::reduce_ker_im_cok<Int, Real>(K, L, IdMapping, params));
    if (params.kernel) KICR.kernel = true;
    if (params.image) KICR.image = true;
    if (params.cokernel) KICR.cokernel = true;

    return KICR;
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

    using IndexDiagram = PyOineusDiagrams<size_t>;
    using IndexDgmPtVec = typename oin::Diagrams<size_t>::Dgm;

    std::string vr_edge_name = "VREdge";

    std::string index_dgm_class_name = "IndexDiagrams";

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
            .def_readwrite("kernel", &ReductionParams::kernel)
            .def_readwrite("image", &ReductionParams::image)
            .def_readwrite("cokernel", &ReductionParams::cokernel)
            .def_readwrite("verbose", &ReductionParams::verbose)
            .def(py::pickle(
                    // __getstate__

                    [](const ReductionParams& p) {
                      return py::make_tuple(p.n_threads, p.chunk_size, p.write_dgms,
                              p.sort_dgms, p.clearing_opt, p.acq_rel, p.print_time, p.compute_v, p.compute_u,
                              p.do_sanity_check, p.elapsed, p.kernel, p.image, p.cokernel, p.verbose);
                    },
                    // __setstate__
                    [](py::tuple t) {
                      if (t.size() != 15)
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
                      p.kernel = t[i++].cast<decltype(p.kernel)>();
                      p.image = t[i++].cast<decltype(p.image)>();
                      p.cokernel = t[i++].cast<decltype(p.cokernel)>();
                      p.verbose = t[i++].cast<decltype(p.verbose)>();

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

    using ReductionParams = oin::Params;

    using IndexDiagram = PyOineusDiagrams<size_t>;
    using IndexDgmPtVec = typename oin::Diagrams<size_t>::Dgm;

    std::string vr_edge_name = "VREdge";

    std::string index_dgm_class_name = "IndexDiagrams";

    py::class_<Decomposition>(m, "Decomposition")
            .def(py::init<const oin::Filtration<oin::Simplex<Int, double>>&, bool>())
            .def(py::init<const oin::Filtration<oin::Simplex<Int, float>>&, bool>())
            .def_readwrite("r_data", &Decomposition::r_data)
            .def_readwrite("v_data", &Decomposition::v_data)
            .def_readwrite("u_data_t", &Decomposition::u_data_t)
            .def_readwrite("d_data", &Decomposition::d_data)
            .def("reduce", &Decomposition::reduce, py::call_guard<py::gil_scoped_release>())
            .def("sanity_check", &Decomposition::sanity_check, py::call_guard<py::gil_scoped_release>())
            .def("diagram", [](const Decomposition& self, const oin::Filtration<oin::Simplex<Int, double>>& fil, bool include_inf_points) { return PyOineusDiagrams<double>(self.diagram(fil, include_inf_points)); })
            .def("diagram", [](const Decomposition& self, const oin::Filtration<oin::Simplex<Int, float>>& fil, bool include_inf_points) { return PyOineusDiagrams<float>(self.diagram(fil, include_inf_points)); })
            .def("zero_persistence_diagram", [](const Decomposition& self, const oin::Filtration<oin::Simplex<Int, float>>& fil) { return PyOineusDiagrams<float>(self.zero_persistence_diagram(fil)); })
            .def("index_diagram", [](const Decomposition& self, const oin::Filtration<oin::Simplex<Int, double>>& fil, bool include_inf_points, bool include_zero_persistence_points) {
              return PyOineusDiagrams<size_t>(self.index_diagram(fil, include_inf_points, include_zero_persistence_points));
            })
            .def("index_diagram", [](const Decomposition& self, const oin::Filtration<oin::Simplex<Int, float>>& fil, bool include_inf_points, bool include_zero_persistence_points) {
              return PyOineusDiagrams<size_t>(self.index_diagram(fil, include_inf_points, include_zero_persistence_points));
            });
}

void init_oineus_common_decomposition_int(py::module& m);

template<class Int>
void init_oineus_common_diagram(py::module& m)
{
    using namespace pybind11::literals;

    using oin::VREdge;
    using oin::DenoiseStrategy;
    using oin::ConflictStrategy;

    using IndexDiagram = PyOineusDiagrams<size_t>;
    using IndexDgmPtVec = typename oin::Diagrams<size_t>::Dgm;

    std::string index_dgm_class_name = "IndexDiagrams";

    using DgmPointInt = typename oin::DgmPoint<Int>;
    using DgmPointSizet = typename oin::DgmPoint<size_t>;

    py::class_<DgmPointInt>(m, "DgmPoint_int")
            .def(py::init<Int, Int>())
            .def_readwrite("birth", &DgmPointInt::birth)
            .def_readwrite("death", &DgmPointInt::death)
            .def("__getitem__", [](const DgmPointInt& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPointInt& p) { return std::hash<DgmPointInt>()(p); })
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
            .def("__repr__", [](const DgmPointSizet& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<IndexDiagram>(m, index_dgm_class_name.c_str())
            .def(py::init<dim_type>())
            .def("in_dimension", [](const IndexDiagram& self, dim_type dim, bool as_numpy) -> std::variant<pybind11::array_t<typename IndexDiagram::Coordinate>, IndexDgmPtVec> {
                      if (as_numpy)
                          return self.get_diagram_in_dimension_as_numpy(dim);
                      else
                          return self.get_diagram_in_dimension(dim);
                    }, "return persistence diagram in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                    py::arg("dim"), py::arg("as_numpy")=true)
            .def("__getitem__", &IndexDiagram::get_diagram_in_dimension);
}

void init_oineus_common_diagram_int(py::module& m);

template<class Int, class Real>
void init_oineus_functions(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using Filtration = oin::Filtration<oin::Simplex<Int, Real>>;
    using Simplex = typename Filtration::Cell;

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
    m.def(func_name.c_str(), &oin::get_denoise_target<Simplex>);

    func_name = "get_wasserstein_matching_target_values" + suffix;
    m.def(func_name.c_str(), &oin::get_target_from_matching<Simplex>);

    // target values -- diagram loss
    func_name = "get_target_values_diagram_loss" + suffix;
    m.def(func_name.c_str(), &oin::get_prescribed_simplex_values_diagram_loss<Real>);

    // target values --- X set
    func_name = "get_target_values_x" + suffix;
    m.def(func_name.c_str(), &oin::get_prescribed_simplex_values_set_x<Simplex>);

    // to reproduce "Topology layer for ML" experiments
    func_name = "get_bruelle_target" + suffix;
    m.def(func_name.c_str(), &oin::get_bruelle_target<Simplex>);

    // to reproduce "Well group loss" experiments
    func_name = "get_well_group_target" + suffix;
    m.def(func_name.c_str(), &oin::get_well_group_target<Simplex>);

    func_name = "get_nth_persistence" + suffix;
    m.def(func_name.c_str(), &oin::get_nth_persistence<Simplex>);

    // to equidistribute points
    func_name = "get_barycenter_target" + suffix;
    m.def(func_name.c_str(), &oin::get_barycenter_target<Simplex>);

    // to get permutation for Warm Starts
    func_name = "get_permutation" + suffix;
    m.def(func_name.c_str(), &oin::targets_to_permutation<Simplex>);

    func_name = "get_permutation_dtv" + suffix;
    m.def(func_name.c_str(), &oin::targets_to_permutation_dtv<Simplex>);

    // reduce to create an ImKerReduced object
    func_name = "reduce_ker_im_cok" + suffix;
    m.def(func_name.c_str(), &oin::reduce_ker_im_cok<Int, Real>);

    func_name = "list_to_filtration" + suffix;
    m.def(func_name.c_str(), &list_to_filtration<Int, Real>);

    func_name = "compute_kernel_image_cokernel_reduction" + suffix;
    m.def(func_name.c_str(), &compute_kernel_image_cokernel_reduction<Int, Real>);

    func_name = "get_ls_filtration" + suffix;
    m.def(func_name.c_str(), &get_ls_filtration<Int, Real>);
}

void init_oineus_functions_double(py::module& m, std::string suffix);
void init_oineus_functions_float(py::module& m, std::string suffix);

template<class Int, class Real>
void init_oineus_fil_dgm_simplex(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using DgmPoint = typename oin::Diagrams<Real>::Point;
    using DgmPtVec = typename oin::Diagrams<Real>::Dgm;
    using Diagram = PyOineusDiagrams<Real>;


    using Filtration = oin::Filtration<oin::Simplex<Int, Real>>;
    using Simplex = typename Filtration::Cell;

    using oin::VREdge;

    using VRUDecomp = oin::VRUDecomposition<Int>;
    using KerImCokRed = oin::KerImCokReduced<Int, Real>;
    using PyKerImCokRed = PyKerImCokRed<Int, Real>;
    //using CokRed =  oin::CokReduced<Int, Real>;

    std::string filtration_class_name = "Filtration" + suffix;
    std::string simplex_class_name = "Simplex" + suffix;

    std::string dgm_point_name = "DiagramPoint" + suffix;
    std::string dgm_class_name = "Diagrams" + suffix;

    std::string ker_im_cok_reduced_class_name = "KerImCokReduced" + suffix;
    std::string py_ker_im_cok_reduced_class_name = "PyKerImCokRed" + suffix;

    py::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<Real, Real>(), py::arg("birth"), py::arg("death"))
            .def_readwrite("birth", &DgmPoint::birth)
            .def_readwrite("death", &DgmPoint::death)
            .def("__getitem__", [](const DgmPoint& p, int i) { return p[i]; })
            .def("__hash__", [](const DgmPoint& p) { return std::hash<DgmPoint>()(p); })
            .def("__repr__", [](const DgmPoint& p) {
              std::stringstream ss;
              ss << p;
              return ss.str();
            });

    py::class_<Diagram>(m, dgm_class_name.c_str())
            .def(py::init<dim_type>())
            .def("in_dimension", [](const Diagram& self, dim_type dim, bool as_numpy) -> std::variant<pybind11::array_t<Real>, DgmPtVec> {
                if (as_numpy)
                    return self.get_diagram_in_dimension_as_numpy(dim);
                else
                    return self.get_diagram_in_dimension(dim);
                }, "return persistence diagram in dimension dim: if as_numpy is False (default), the diagram is returned as list of DgmPoints, else as NumPy array",
                        py::arg("dim"), py::arg("as_numpy")=true)
            .def("__getitem__", &Diagram::get_diagram_in_dimension_as_numpy);

    py::class_<Simplex>(m, simplex_class_name.c_str())
            .def(py::init<typename Simplex::IdxVector, Real>(), py::arg("vertices"), py::arg("value"))
            .def(py::init<typename Simplex::Int, typename Simplex::IdxVector, Real>(), py::arg("id"), py::arg("vertices"), py::arg("value"))
            .def_readwrite("id", &Simplex::id_)
            .def_readwrite("sorted_id", &Simplex::sorted_id_)
            .def_readwrite("vertices", &Simplex::vertices_)
            .def_readwrite("value", &Simplex::value_)
            .def("dim", &Simplex::dim)
            .def("boundary", &Simplex::boundary)
            .def("join", [](const Simplex& sigma, Int new_vertex, Real value, Int new_id) {
                    return sigma.join(new_id, new_vertex, value);
                },
                py::arg("new_vertex"),
                py::arg("value"),
                py::arg("new_id") = Simplex::k_invalid_id)
            .def("__repr__", [](const Simplex& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });

    py::class_<Filtration>(m, filtration_class_name.c_str())
            .def(py::init<typename Filtration::CellVector, bool, int, bool, bool>(),
                    py::arg("cells"),
                    py::arg("negate")=false,
                    py::arg("n_threads")=1,
                    py::arg("sort_only_by_dimension")=false,
                    py::arg("set_ids")=true)
            .def("max_dim", &Filtration::max_dim)
            .def("cells", &Filtration::cells_copy)
            .def("simplices", &Filtration::cells_copy)
            .def("size", &Filtration::size)
            .def("__len__", &Filtration::size)
            .def("size_in_dimension", &Filtration::size_in_dimension)
            .def("n_vertices", &Filtration::n_vertices)
            .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, py::arg("sorted_id"))
            .def("get_id_by_sorted_id", &Filtration::get_id_by_sorted_id, py::arg("sorted_id"))
            .def("get_sorted_id_by_id", &Filtration::get_sorted_id, py::arg("id"))
            .def("get_sorting_permutation", &Filtration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .def("simplex_value_by_vertices", &Filtration::value_by_vertices, py::arg("vertices"))
            .def("get_sorted_id_by_vertices", &Filtration::get_sorted_id_by_vertices, py::arg("vertices"))
            .def("boundary_matrix", &Filtration::boundary_matrix_full)
            .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            .def("__repr__", [](const Filtration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            });

    py::class_<KerImCokRed>(m, ker_im_cok_reduced_class_name.c_str())
            .def(py::init<Filtration, Filtration, VRUDecomp, VRUDecomp, VRUDecomp, VRUDecomp, VRUDecomp, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, oin::Params>());

	  py::class_<PyKerImCokRed>(m, py_ker_im_cok_reduced_class_name.c_str())
            .def(py::init<KerImCokRed>())
            .def("kernel_diagrams", &PyKerImCokRed::kernel_diagrams)
            .def("image_diagrams", &PyKerImCokRed::image_diagrams)
            .def("cokernel_diagrams", &PyKerImCokRed::cokernel_diagrams)
            .def("D_F", &PyKerImCokRed::D_F)
            .def("D_G", &PyKerImCokRed::D_G)
            .def("D_Ker", &PyKerImCokRed::D_Ker)
            .def("D_Im", &PyKerImCokRed::D_Im)
            .def("D_Cok", &PyKerImCokRed::D_Cok)
            .def("R_F", &PyKerImCokRed::R_F)
            .def("R_G", &PyKerImCokRed::R_G)
            .def("R_Ker", &PyKerImCokRed::R_Ker)
            .def("R_Im", &PyKerImCokRed::R_Im)
            .def("R_Cok", &PyKerImCokRed::R_Cok)
            .def("V_F", &PyKerImCokRed::V_F)
            .def("V_G", &PyKerImCokRed::V_G)
            .def("V_Ker", &PyKerImCokRed::V_Ker)
            .def("V_Im", &PyKerImCokRed::V_Im)
            .def("V_Cok", &PyKerImCokRed::V_Cok);
}

void init_oineus_fil_dgm_simplex_float(py::module& m, std::string suffix);
void init_oineus_fil_dgm_simplex_double(py::module& m, std::string suffix);

void init_oineus_top_optimizer(py::module& m);


#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
