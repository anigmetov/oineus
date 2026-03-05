#include <sstream>
#include <string>
#include <vector>

#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include <oineus/common_defs.h>
#include <oineus/params.h>
#include <oineus/simplex.h>
#include <oineus/cell_with_value.h>
#include <oineus/filtration.h>
#include <oineus/diagram.h>
#include <oineus/decomposition.h>

namespace {

using OinInt = long int;
using OinReal = double;

using VREdge = oineus::VREdge<OinInt>;
using ReductionParams = oineus::Params;

using CombinatorialSimplex = oineus::Simplex<OinInt>;
using Simplex = oineus::CellWithValue<CombinatorialSimplex, OinReal>;
using Filtration = oineus::Filtration<CombinatorialSimplex, OinReal>;

using Decomposition = oineus::VRUDecomposition<OinInt>;
using MatrixData = typename Decomposition::MatrixData;

using DiagramPoint = oineus::DgmPoint<OinReal>;
using IndexDiagramPoint = oineus::DgmPoint<size_t>;
using Diagrams = oineus::Diagrams<OinReal>;

VREdge make_vr_edge(OinInt x, OinInt y)
{
    return VREdge {x, y};
}

std::string vr_edge_repr(const VREdge& edge)
{
    std::stringstream ss;
    ss << edge;
    return ss.str();
}

Simplex make_simplex(const std::vector<OinInt>& vertices, OinReal value)
{
    return {CombinatorialSimplex(vertices), value};
}

Simplex make_simplex_with_id(OinInt id, const std::vector<OinInt>& vertices, OinReal value)
{
    return {CombinatorialSimplex(id, vertices), value};
}

Filtration make_filtration(const std::vector<Simplex>& simplices, bool negate, int n_threads)
{
    return {simplices, negate, n_threads};
}

std::vector<OinInt> simplex_vertices(const Simplex& sigma)
{
    return sigma.get_cell().get_vertices();
}

std::string simplex_repr(const Simplex& sigma)
{
    return sigma.repr();
}

std::string combinatorial_simplex_repr(const CombinatorialSimplex& sigma)
{
    return sigma.repr();
}

std::string reduction_params_repr(const ReductionParams& params)
{
    std::stringstream ss;
    ss << params;
    return ss.str();
}

std::vector<DiagramPoint> diagram_points_in_dimension(Diagrams& dgms, size_t dim)
{
    return dgms.get_diagram_in_dimension(dim);
}

std::vector<IndexDiagramPoint> index_diagram_points_in_dimension(const Diagrams& dgms, size_t dim)
{
    return dgms.get_index_diagram_in_dimension(dim);
}

} // namespace

namespace jlcxx {

template<>
struct IsMirroredType<oineus::VREdge<long int>> : std::false_type { };

template<>
struct IsMirroredType<oineus::Params> : std::false_type { };

template<>
struct IsMirroredType<oineus::DgmPoint<double>> : std::false_type { };

template<>
struct IsMirroredType<oineus::DgmPoint<size_t>> : std::false_type { };

} // namespace jlcxx

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.add_type<VREdge>("VREdge")
            .constructor<>()
            .method("x", [](const VREdge& edge) { return edge.x; })
            .method("set_x!", [](VREdge& edge, OinInt x) { edge.x = x; })
            .method("y", [](const VREdge& edge) { return edge.y; })
            .method("set_y!", [](VREdge& edge, OinInt y) { edge.y = y; })
            .method("repr", [](const VREdge& edge) { return vr_edge_repr(edge); });

    mod.add_type<ReductionParams>("ReductionParams")
            .constructor<>()
            .method("n_threads", [](const ReductionParams& p) { return p.n_threads; })
            .method("set_n_threads!", [](ReductionParams& p, int value) { p.n_threads = value; })
            .method("chunk_size", [](const ReductionParams& p) { return p.chunk_size; })
            .method("set_chunk_size!", [](ReductionParams& p, int value) { p.chunk_size = value; })
            .method("clearing_opt", [](const ReductionParams& p) { return p.clearing_opt; })
            .method("set_clearing_opt!", [](ReductionParams& p, bool value) { p.clearing_opt = value; })
            .method("compute_v", [](const ReductionParams& p) { return p.compute_v; })
            .method("set_compute_v!", [](ReductionParams& p, bool value) { p.compute_v = value; })
            .method("compute_u", [](const ReductionParams& p) { return p.compute_u; })
            .method("set_compute_u!", [](ReductionParams& p, bool value) { p.compute_u = value; })
            .method("restore_elz", [](const ReductionParams& p) { return p.restore_elz; })
            .method("set_restore_elz!", [](ReductionParams& p, bool value) { p.restore_elz = value; })
            .method("sort_dgms", [](const ReductionParams& p) { return p.sort_dgms; })
            .method("set_sort_dgms!", [](ReductionParams& p, bool value) { p.sort_dgms = value; })
            .method("verbose", [](const ReductionParams& p) { return p.verbose; })
            .method("set_verbose!", [](ReductionParams& p, bool value) { p.verbose = value; })
            .method("repr", [](const ReductionParams& p) { return reduction_params_repr(p); });

    mod.add_type<CombinatorialSimplex>("CombinatorialSimplex")
            .constructor<>()
            .constructor<const std::vector<OinInt>&>()
            .constructor<OinInt, const std::vector<OinInt>&>()
            .method("id", &CombinatorialSimplex::get_id)
            .method("set_id!", &CombinatorialSimplex::set_id)
            .method("uid", &CombinatorialSimplex::get_uid)
            .method("dim", &CombinatorialSimplex::dim)
            .method("vertices", &CombinatorialSimplex::get_vertices)
            .method("boundary", &CombinatorialSimplex::boundary)
            .method("join", [](const CombinatorialSimplex& sigma, OinInt new_vertex, OinInt new_id) { return sigma.join(new_id, new_vertex); })
            .method("repr", [](const CombinatorialSimplex& sigma) { return combinatorial_simplex_repr(sigma); });

    mod.add_type<Simplex>("Simplex")
            .constructor<>()
            .constructor<const CombinatorialSimplex&, OinReal>()
            .method("id", &Simplex::get_id)
            .method("set_id!", &Simplex::set_id)
            .method("uid", &Simplex::get_uid)
            .method("dim", &Simplex::dim)
            .method("value", &Simplex::get_value)
            .method("set_value!", &Simplex::set_value)
            .method("sorted_id", &Simplex::get_sorted_id)
            .method("set_sorted_id!", &Simplex::set_sorted_id)
            .method("vertices", [](const Simplex& sigma) { return simplex_vertices(sigma); })
            .method("boundary", &Simplex::boundary)
            .method("combinatorial_simplex", &Simplex::get_cell)
            .method("join", [](const Simplex& sigma, OinInt new_vertex, OinReal value, OinInt new_id) {
                return sigma.join(new_id, new_vertex, value);
            })
            .method("repr", [](const Simplex& sigma) { return simplex_repr(sigma); });

    mod.add_type<Filtration>("Filtration")
            .constructor<>()
            .constructor<const std::vector<Simplex>&, bool, int>()
            .method("size", &Filtration::size)
            .method("max_dim", &Filtration::max_dim)
            .method("size_in_dimension", &Filtration::size_in_dimension)
            .method("n_vertices", &Filtration::n_vertices)
            .method("negate", &Filtration::negate)
            .method("cells", &Filtration::cells_copy)
            .method("cell", &Filtration::get_cell)
            .method("value_by_sorted_id", &Filtration::value_by_sorted_id)
            .method("id_by_sorted_id", &Filtration::get_id_by_sorted_id)
            .method("sorted_id_by_id", &Filtration::get_sorted_id)
            .method("sorting_permutation", &Filtration::get_sorting_permutation)
            .method("inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .method("boundary_matrix", [](const Filtration& fil, int n_threads) { return fil.boundary_matrix(n_threads); })
            .method("boundary_matrix", [](const Filtration& fil) { return fil.boundary_matrix(1); })
            .method("coboundary_matrix", [](const Filtration& fil, int n_threads) { return fil.coboundary_matrix(n_threads); })
            .method("coboundary_matrix", [](const Filtration& fil) { return fil.coboundary_matrix(1); });

    mod.add_type<DiagramPoint>("DiagramPoint")
            .constructor<>()
            .constructor<OinReal, OinReal>()
            .constructor<OinReal, OinReal, size_t, size_t>()
            .method("birth", [](const DiagramPoint& p) { return p.birth; })
            .method("death", [](const DiagramPoint& p) { return p.death; })
            .method("birth_index", [](const DiagramPoint& p) { return p.birth_index; })
            .method("death_index", [](const DiagramPoint& p) { return p.death_index; })
            .method("persistence", &DiagramPoint::persistence)
            .method("is_inf", &DiagramPoint::is_inf)
            .method("is_diagonal", &DiagramPoint::is_diagonal);

    mod.add_type<IndexDiagramPoint>("IndexDiagramPoint")
            .constructor<>()
            .constructor<size_t, size_t>()
            .constructor<size_t, size_t, size_t, size_t>()
            .method("birth", [](const IndexDiagramPoint& p) { return p.birth; })
            .method("death", [](const IndexDiagramPoint& p) { return p.death; })
            .method("birth_index", [](const IndexDiagramPoint& p) { return p.birth_index; })
            .method("death_index", [](const IndexDiagramPoint& p) { return p.death_index; });

    mod.add_type<Diagrams>("Diagrams")
            .constructor<>()
            .constructor<oineus::dim_type>()
            .method("n_dims", &Diagrams::n_dims)
            .method("diagram_in_dimension", [](Diagrams& d, size_t dim) { return diagram_points_in_dimension(d, dim); })
            .method("index_diagram_in_dimension", [](const Diagrams& d, size_t dim) { return index_diagram_points_in_dimension(d, dim); })
            .method("sort!", &Diagrams::sort);

    mod.add_type<Decomposition>("Decomposition")
            .constructor<>()
            .constructor<const Filtration&, bool, int>()
            .constructor<const MatrixData&, size_t, bool, bool>()
            .method("size", &Decomposition::size)
            .method("dualize", &Decomposition::dualize)
            .method("d_data", [](const Decomposition& d) { return d.d_data; })
            .method("r_data", [](const Decomposition& d) { return d.r_data; })
            .method("v_data", [](const Decomposition& d) { return d.v_data; })
            .method("u_data_t", [](const Decomposition& d) { return d.u_data_t; })
            .method("dim_first", [](const Decomposition& d) { return d.dim_first; })
            .method("dim_last", [](const Decomposition& d) { return d.dim_last; })
            .method("reduce!", [](Decomposition& d, ReductionParams& p) { d.reduce(p); })
            .method("reduce!", [](Decomposition& d) {
                ReductionParams p;
                d.reduce(p);
            })
            .method("diagram", [](const Decomposition& d, const Filtration& fil, bool include_inf_points) {
                return d.diagram(fil, include_inf_points);
            })
            .method("diagram", [](const Decomposition& d, const Filtration& fil) {
                return d.diagram(fil, true);
            });

    mod.method("make_vr_edge", &make_vr_edge);
    mod.method("make_simplex", &make_simplex);
    mod.method("make_simplex_with_id", &make_simplex_with_id);
    mod.method("make_filtration", &make_filtration);
}
