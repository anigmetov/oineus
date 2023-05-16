#ifndef OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
#define OINEUS_OINEUS_PERSISTENCE_BINDINGS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <oineus/timer.h>
#include <oineus/oineus.h>

using dim_type = oineus::dim_type;

template<class Real>
class PyOineusDiagrams {
public:
    PyOineusDiagrams() = default;

    PyOineusDiagrams(const oineus::Diagrams<Real>& _diagrams)
            :diagrams_(_diagrams) { }

    PyOineusDiagrams(oineus::Diagrams<Real>&& _diagrams)
            :diagrams_(_diagrams) { }

    py::array_t<Real> get_diagram_in_dimension(dim_type d)
    {
        auto dgm = diagrams_.get_diagram_in_dimension(d);

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

private:
    oineus::Diagrams<Real> diagrams_;
};

template<class Int, class Real>
using DiagramV = std::pair<PyOineusDiagrams<Real>, typename oineus::VRUDecomposition<Int>::MatrixData>;

template<class Int, class Real>
using DiagramRV = std::tuple<PyOineusDiagrams<Real>, typename oineus::VRUDecomposition<Int>::MatrixData, typename oineus::VRUDecomposition<Int>::MatrixData>;

template<class Int, class Real, size_t D>
typename oineus::Grid<Int, Real, D>
get_grid(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool wrap)
{
    using Grid = oineus::Grid<Int, Real, D>;
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

template<class Real, size_t D>
decltype(auto) numpy_to_point_vector(py::array_t<Real, py::array::c_style | py::array::forcecast> data)
{
    using PointVector = std::vector<oineus::Point<Real, D>>;

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
typename oineus::Filtration<oineus::Simplex<Int, Real>>
list_to_filtration(py::list data, oineus::Params& params) //take a list of simplices and turn it into a filtration for oineus. The list should contain simplices in the form '[id, [boundary], filtration value]'. 
{
    using IdxVector = std::vector<Int>;
    using Simplex = oineus::Simplex<Int, Real>;
    using Filtration = oineus::Filtration<Simplex>;
    using FiltrationSimplexVector = typename Filtration::SimplexVector;

    int n_simps = data.size();
    std::cout << "Number of cells in the complex is: " << n_simps << std::endl;
    FiltrationSimplexVector FSV;

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
    Filtration Filt(FSV, false, 1);

    return Filt;
}

template<typename Int, typename Real>
decltype(auto)
get_ls_filtration(const py::list& simplices, const py::array_t<Real>& vertex_values, bool negate, int n_threads)
// take a list of simplices and a numpy array of their values and turn it into a filtration for oineus.
// The list should contain simplices, each simplex is a list of vertices,
// e.g., triangulation of one segment is [[0], [1], [0, 1]]
{
    using Simplex = oineus::Simplex<Int, Real>;
    using Filtration = oineus::Filtration<Simplex>;
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

    return Filtration(std::move(fil_simplices), negate, n_threads);
}

template<class Int, class Real, size_t D>
decltype(auto)
get_vr_filtration(py::array_t<Real, py::array::c_style | py::array::forcecast> points, dim_type max_dim, Real max_radius, int n_threads)
{
    return oineus::get_vr_filtration_bk<Int, Real, D>(numpy_to_point_vector<Real, D>(points), max_dim, max_radius, n_threads);
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
typename oineus::VRUDecomposition<Int>::MatrixData
get_boundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads).first;
    return fil.boundary_matrix_full();
}

template<class Int, class Real, size_t D>
typename oineus::VRUDecomposition<Int>::MatrixData
get_coboundary_matrix(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, int n_threads)
{
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim, n_threads).first;
    auto bm = fil.boundary_matrix_full();
    return oineus::antitranspose(bm);
}

template<class Int, class Real, size_t D>
PyOineusDiagrams<Real>
compute_diagrams_ls_freudenthal(py::array_t<Real, py::array::c_style | py::array::forcecast> data, bool negate, bool wrap, dim_type max_dim, oineus::Params& params, bool include_inf_points)
{
    // for diagram in dimension d, we need (d+1)-simplices
    Timer timer;
    auto fil = get_fr_filtration<Int, Real, D>(data, negate, wrap, max_dim + 1, params.n_threads).first;
    auto elapsed_fil = timer.elapsed_reset();
    oineus::VRUDecomposition<Int> decmp {fil, false};
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
    oineus::KerImCokReduced<Int, Real> KICR;
public:

    PyKerImCokDgms(oineus::KerImCokReduced<Int, Real> KICR_)
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
		oineus::KerImCokReduced<Int, Real> KICR;

public:
    bool kernel {false};
    bool image {false};
    bool cokernel {false};

    PyKerImCokRed(oineus::KerImCokReduced<Int, Real> KICR_)
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
        oineus::Params& params) //take a list of simplices and turn it into a filtration for oineus. The list should contain simplices in the form '[id, [boundary], filtration value].
{
    using IdxVector = std::vector<Int>;
    using IntSparseColumn = oineus::SparseColumn<Int>;
    using MatrixData = std::vector<IntSparseColumn>;
    using FiltrationSimplex = oineus::Simplex<Int, Real>;
    using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
    using Filtration = oineus::Filtration<FiltrationSimplex>;
    using KerImCokReduced = oineus::KerImCokReduced<Int, Real>;
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

    PyKerImCokRed KICR(oineus::reduce_ker_im_cok<Int, Real>(K, L, IdMapping, params));
    if (params.kernel) KICR.kernel = true;
    if (params.image) KICR.image = true;
    if (params.cokernel) KICR.cokernel = true;

    return KICR;
}

template<class Int>
void init_oineus_common(py::module& m)
{
    using namespace pybind11::literals;

    using oineus::VREdge;

    using oineus::DenoiseStrategy;
    using oineus::ConflictStrategy;

    using Decomposition = oineus::VRUDecomposition<Int>;

    using ReductionParams = oineus::Params;

    using IndexDiagram = PyOineusDiagrams<size_t>;

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
            .value("FixCritAvg", ConflictStrategy::FixCritAvg, "use matching on critical, average gradients on other simplices")
            .def("as_str", [](const ConflictStrategy& self) { return conflict_strategy_to_string(self); });

    py::class_<Decomposition>(m, "Decomposition")
            .def(py::init<const oineus::Filtration<oineus::Simplex<Int, double>>&, bool>())
            .def(py::init<const oineus::Filtration<oineus::Simplex<Int, float>>&, bool>())
            .def_readwrite("r_data", &Decomposition::r_data)
            .def_readwrite("v_data", &Decomposition::v_data)
            .def_readwrite("u_data_t", &Decomposition::u_data_t)
            .def_readwrite("d_data", &Decomposition::d_data)
            .def("reduce", &Decomposition::reduce, py::call_guard<py::gil_scoped_release>())
            .def("sanity_check", &Decomposition::sanity_check, py::call_guard<py::gil_scoped_release>())
            .def("diagram", [](const Decomposition& self, const oineus::Filtration<oineus::Simplex<Int, double>>& fil, bool include_inf_points) { return PyOineusDiagrams<double>(self.diagram(fil, include_inf_points)); })
            .def("diagram", [](const Decomposition& self, const oineus::Filtration<oineus::Simplex<Int, float>>& fil, bool include_inf_points) { return PyOineusDiagrams<float>(self.diagram(fil, include_inf_points)); })
            .def("index_diagram", [](const Decomposition& self, const oineus::Filtration<oineus::Simplex<Int, double>>& fil, bool include_inf_points, bool include_zero_persistence_points) {
              return PyOineusDiagrams<size_t>(self.index_diagram(fil, include_inf_points, include_zero_persistence_points));
            })
            .def("index_diagram", [](const Decomposition& self, const oineus::Filtration<oineus::Simplex<Int, float>>& fil, bool include_inf_points, bool include_zero_persistence_points) {
              return PyOineusDiagrams<size_t>(self.index_diagram(fil, include_inf_points, include_zero_persistence_points));
            })
            ;

    using DgmPointInt = typename oineus::DgmPoint<Int>;
    using DgmPointSizet = typename oineus::DgmPoint<size_t>;

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
            .def("in_dimension", &IndexDiagram::get_diagram_in_dimension)
            .def("__getitem__", &IndexDiagram::get_diagram_in_dimension);
}

template<class Int, class Real>
void init_oineus(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using DgmPoint = oineus::DgmPoint<Real>;
    using Diagram = PyOineusDiagrams<Real>;

    using Simplex = oineus::Simplex<Int, Real>;
    using Filtration = oineus::Filtration<Simplex>;

    using oineus::VREdge;

    using VRUDecomp = oineus::VRUDecomposition<Int>;
    using KerImCokRed = oineus::KerImCokReduced<Int, Real>;
    using PyKerImCokRed = PyKerImCokRed<Int, Real>;
    //using CokRed =  oineus::CokReduced<Int, Real>;

    std::string filtration_class_name = "Filtration" + suffix;
    std::string simplex_class_name = "Simplex" + suffix;

    std::string dgm_point_name = "DiagramPoint" + suffix;
    std::string dgm_class_name = "Diagrams" + suffix;

    std::string ker_im_cok_reduced_class_name = "KerImCokReduced" + suffix;
    std::string py_ker_im_cok_reduced_class_name = "PyKerImCokRed" + suffix;

    py::class_<DgmPoint>(m, dgm_point_name.c_str())
            .def(py::init<Real, Real>())
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
            .def("in_dimension", &Diagram::get_diagram_in_dimension)
            .def("__getitem__", &Diagram::get_diagram_in_dimension);

    py::class_<Simplex>(m, simplex_class_name.c_str())
            .def(py::init<typename Simplex::IdxVector, Real>())
            .def_readonly("id", &Simplex::id_)
            .def_readonly("sorted_id", &Simplex::sorted_id_)
            .def_readonly("vertices", &Simplex::vertices_)
            .def_readonly("value", &Simplex::value_)
            .def("dim", &Simplex::dim)
            .def("boundary", &Simplex::boundary)
            .def("__repr__", [](const Simplex& sigma) {
              std::stringstream ss;
              ss << sigma;
              return ss.str();
            });


    py::class_<Filtration>(m, filtration_class_name.c_str())
            .def(py::init<typename Filtration::SimplexVector, bool, int>())
            .def("max_dim", &Filtration::max_dim)
            .def("simplices", &Filtration::simplices_copy)
            .def("size_in_dimension", &Filtration::size_in_dimension)
            .def("simplex_value", &Filtration::value_by_sorted_id)
            .def("boundary_matrix", &Filtration::boundary_matrix_full);


    py::class_<KerImCokRed>(m, ker_im_cok_reduced_class_name.c_str())
			      .def(py::init<oineus::Filtration<oineus::Simplex<Int, Real>>, oineus::Filtration<oineus::Simplex<Int, Real>>, VRUDecomp, VRUDecomp, VRUDecomp, VRUDecomp, VRUDecomp, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, oineus::Params>());

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

    // Vietoris--Rips filtration
    func_name = "get_vr_filtration" + suffix + "_1";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 1>);

    func_name = "get_vr_filtration" + suffix + "_2";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 2>);

    func_name = "get_vr_filtration" + suffix + "_3";
    m.def(func_name.c_str(), &get_vr_filtration<Int, Real, 3>);

    // boundary matrix as vector of columns
    func_name = "get_boundary_matrix" + suffix + "_1";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 1>);

    func_name = "get_boundary_matrix" + suffix + "_2";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 2>);

    func_name = "get_boundary_matrix" + suffix + "_3";
    m.def(func_name.c_str(), &get_boundary_matrix<Int, Real, 3>);

    // target values
    func_name = "get_denoise_target" + suffix;
    m.def(func_name.c_str(), &oineus::get_denoise_target<Int, Real>);

    func_name = "get_wasserstein_matching_target_values" + suffix;
    m.def(func_name.c_str(), &oineus::get_target_from_matching<Int, Real>);

    // target values -- diagram loss
    func_name = "get_target_values_diagram_loss" + suffix;
    m.def(func_name.c_str(), &oineus::get_prescribed_simplex_values_diagram_loss<Real>);

    // target values --- X set
    func_name = "get_target_values_x" + suffix;
    m.def(func_name.c_str(), &oineus::get_prescribed_simplex_values_set_x<Int, Real>);

    // to reproduce "Topology layer for ML" experiments
    func_name = "get_bruelle_target" + suffix;
    m.def(func_name.c_str(), &oineus::get_bruelle_target<Int, Real>);

    // to reproduce "Well group loss" experiments
    func_name = "get_well_group_target" + suffix;
    m.def(func_name.c_str(), &oineus::get_well_group_target<Int, Real>);

    func_name = "get_nth_persistence" + suffix;
    m.def(func_name.c_str(), &oineus::get_nth_persistence<Int, Real>);

    // to equidistribute points
    func_name = "get_barycenter_target" + suffix;
    m.def(func_name.c_str(), &oineus::get_barycenter_target<Int, Real>);

    // to get permutation for Warm Starts
    func_name = "get_permutation" + suffix;
    m.def(func_name.c_str(), &oineus::targets_to_permutation<Int, Real>);

    func_name = "get_permutation_dtv" + suffix;
    m.def(func_name.c_str(), &oineus::targets_to_permutation_dtv<Int, Real>);

    // reduce to create an ImKerReduced object
    func_name = "reduce_ker_im_cok" + suffix;
    m.def(func_name.c_str(), &oineus::reduce_ker_im_cok<Int, Real>);

    func_name = "list_to_filtration" + suffix;
    m.def(func_name.c_str(), &list_to_filtration<Int, Real>);

    func_name = "compute_kernel_image_cokernel_reduction" + suffix;
    m.def(func_name.c_str(), &compute_kernel_image_cokernel_reduction<Int, Real>);

    func_name = "get_ls_filtration" + suffix;
    m.def(func_name.c_str(), &get_ls_filtration<Int, Real>);
}

inline void init_oineus_top_optimizer(py::module& m)
{

    using Real = double;
    using Int = int;

    using TopologyOptimizer = oineus::TopologyOptimizer<Int, Real>;
    using IndicesValues = typename TopologyOptimizer::IndicesValues;
    using CrititcalSet = typename TopologyOptimizer::CriticalSet;
    using Target = typename TopologyOptimizer::Target;
    using Indices = typename TopologyOptimizer::Indices;
    using Values = typename TopologyOptimizer::Values;
    using CriticalSets = typename TopologyOptimizer::CriticalSets;
    using ConflictStrategy = oineus::ConflictStrategy;

    using Simplex = oineus::Simplex<Int, Real>;
    using Filtration = oineus::Filtration<Simplex>;


    // optimization
    py::class_<TopologyOptimizer>(m, "TopologyOptimizer")
            .def(py::init<const Filtration& >())
            .def("compute_diagram", &TopologyOptimizer::compute_diagram)
            .def("simplify", &TopologyOptimizer::simplify)
            .def("match", &TopologyOptimizer::match)
            .def("singleton", &TopologyOptimizer::singleton)
            .def("singletons", &TopologyOptimizer::singletons)
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, ConflictStrategy)>(&TopologyOptimizer::combine_loss))
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const CriticalSets&, const Target&, ConflictStrategy)>(&TopologyOptimizer::combine_loss))
            .def("combine_loss", static_cast<IndicesValues (TopologyOptimizer::*)(const Indices&, const Values&, ConflictStrategy)>(&TopologyOptimizer::combine_loss))
            .def("update", &TopologyOptimizer::update)
            ;

//    IndicesValues combine_loss(const CriticalSets& critical_sets, ConflictStrategy strategy)
}

#endif //OINEUS_OINEUS_PERSISTENCE_BINDINGS_H
