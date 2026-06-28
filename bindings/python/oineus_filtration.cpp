#include <functional>
#include "oineus_persistence_bindings.h"
#include "oineus_type_list.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/unordered_map.h>


// True for the simplicial cell encodings (fat / Freudenthal anchor+type / bit-packed),
// i.e. Simplex<Int, Enc>. Used to gate the simplex-vertex extractors (get_edges etc.),
// which are meaningless for cubical cells.
template<class C> struct is_simplex_cell : std::false_type {};
template<class Int, class Enc> struct is_simplex_cell<oin::Simplex<Int, Enc>> : std::true_type {};

// Extract the (simplex_dim)-simplices of a simplicial filtration as an (n, simplex_dim+1)
// numpy array of vertex ids. Works for any simplicial encoding: a fat (NoGeometry) cell exposes
// its vertices directly; a slim/packed cell materializes them from the shared geometry. The
// rows are in filtration order within the dimension, so a caller can compute per-simplex values
// (edge lengths, circumradii, ...) aligned to the cells.
template<class Fil>
nb::ndarray<size_t, nb::numpy> extract_simplices_as_numpy(const Fil& fil, dim_type simplex_dim)
{
    using VertexIndex = size_t;
    using UnderCell = std::decay_t<decltype(fil.cells()[0].get_cell())>;
    const dim_type simplex_size = simplex_dim + 1;
    const VertexIndex n_simplices = fil.size_in_dimension(simplex_dim);
    const auto first = fil.dim_first(simplex_dim);

    auto* simplices = new VertexIndex[simplex_size * n_simplices];

    for(auto simplex_idx = first; simplex_idx <= fil.dim_last(simplex_dim); ++simplex_idx) {
        size_t array_idx = simplex_idx - first;
        const UnderCell& under = fil.cells()[simplex_idx].get_cell();
        auto write = [&](const auto& vs) {
            assert(vs.size() == static_cast<size_t>(simplex_size));
            for(size_t v_idx = 0; v_idx < static_cast<size_t>(simplex_size); ++v_idx)
                simplices[simplex_size * array_idx + v_idx] = static_cast<VertexIndex>(vs[v_idx]);
        };
        if constexpr (std::is_same_v<typename UnderCell::Geometry, oin::NoGeometry>)
            write(under.get_vertices());
        else
            write(under.vertices(fil.geometry()));
    }

    nb::capsule free_when_done(simplices, [](void* p) noexcept {
     auto* pp = reinterpret_cast<VertexIndex*>(p);
     delete[] pp;
   });

    return nb::ndarray<VertexIndex, nb::numpy>(simplices, {n_simplices, static_cast<size_t>(simplex_size)}, free_when_done);
}

// ============ Slim filtration bindings (Cube / Freudenthal / bit-packed) ============
// These three filtrations all follow ONE pattern: store slim cells + a shared geometry,
// and materialize an honest fat cell on every Python access. They used to be three
// near-identical hand macros (BIND_CUBE/FR/PACKED_FILTRATION); register_slim_filtration<Policy>
// below binds the ~35 shared methods ONCE, and a per-policy SlimFilTraits<Policy> supplies the
// few DIVERGENT hooks (the fat<->slim conversion family, the uid translation, the pickle state,
// the display words, and the cube-only public ctor + boundary_matrix_rel). for_each_type then
// instantiates it for all eight slim cell types. The fat Simplex/ProdFiltration bindings are a
// different (self-contained, no-geometry) pattern and stay hand-written below.

template<class Policy>
struct SlimFilTraits;   // primary template intentionally undefined: each slim cell specializes it

// --- cubical: slim Cube<Int,D> materializes a FatCube; its uid IS a dense int, so the uid
// accessors take that int directly (no combinatorial translation). Cube alone has a public
// ctor-from-fat-cells and boundary_matrix_rel. Geometry is the GridDomain, stored verbatim. ---
template<unsigned D>
struct SlimFilTraits<oin::Cube<oin_int, D>> {
    using Cell = oin::Cube<oin_int, D>;
    using Fil = oin::Filtration<Cell, oin_real>;
    using SlimValue = oin::CellWithValue<Cell, oin_real>;
    using FatValue = oin::CellWithValue<oin::FatCube<oin_int, D>, oin_real>;
    using Geometry = typename Cell::Geometry;     // GridDomain<oin_int, D>
    using UidArg = oin_int;                        // cube uid == fat uid, a plain int

    static constexpr bool has_ctor = true;
    static constexpr bool has_boundary_matrix_rel = true;
    static std::string py_name() { return "_CubeFiltration_" + std::to_string(D) + "D"; }
    static constexpr const char* cell_word = "cube";
    static constexpr const char* cells_word = "cubes";
    static constexpr const char* value_word = "cube_value_by_sorted_id";

    static FatValue fatten(const SlimValue& cv, const Geometry& g) { return fatten_cube<oin_int, D, oin_real>(cv, g); }
    static std::vector<FatValue> fatten_all_(const Fil& fil) { return fatten_all<oin_int, D, oin_real>(fil); }
    static SlimValue slim(const FatValue& fc, const Geometry&) { return slim_cube<oin_int, D, oin_real>(fc); }
    static typename Cell::Uid to_slim_uid(const Geometry&, UidArg uid) { return uid; }

    using StateTuple = std::tuple<decltype(Fil::negate_), std::vector<FatValue>, decltype(Fil::is_subfiltration_),
            decltype(Fil::uid_to_sorted_id), decltype(Fil::id_to_sorted_id_), decltype(Fil::sorted_id_to_id_),
            decltype(Fil::dim_first_), decltype(Fil::dim_last_), decltype(Fil::kind_), decltype(Fil::geometry_)>;

    static StateTuple getstate(const Fil& fil)
    {
        auto fat = fatten_all<oin_int, D, oin_real>(fil);
        return std::make_tuple(fil.negate_, std::move(fat), fil.is_subfiltration_,
                fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_,
                fil.kind_, fil.geometry_);
    }
    static void setstate(Fil& fil, const StateTuple& t)
    {
        new (&fil) Fil();
        fil.negate_ = std::get<0>(t);
        const auto& fat = std::get<1>(t);
        typename Fil::CellVector slim_cells;
        slim_cells.reserve(fat.size());
        for (const auto& fc : fat) slim_cells.push_back(slim_cube<oin_int, D, oin_real>(fc));
        fil.cells_ = std::move(slim_cells);
        fil.is_subfiltration_ = std::get<2>(t);
        fil.uid_to_sorted_id = std::get<3>(t);
        fil.id_to_sorted_id_ = std::get<4>(t);
        fil.sorted_id_to_id_ = std::get<5>(t);
        fil.dim_first_ = std::get<6>(t);
        fil.dim_last_ = std::get<7>(t);
        fil.kind_ = std::get<8>(t);
        fil.geometry_ = std::get<9>(t);
        // cube uses the flat uid index, not the (empty) hash above; rebuild it from the
        // restored cells (setstate bypasses the ctor)
        fil.rebuild_uid_index_();
    }
};

// --- slim Freudenthal: Simplex<Int, FreudenthalAnchorType<Int,D>> materializes a fat Simplex.
// Factory-produced only (a fat simplex carries no grid domain). The Python-facing uid is the
// universal COMBINATORIAL uid, translated to the slim (anchor,type) uid by fr_slim_uid_from_comb_uid.
// FrGeometry is unpicklable, so the state stores its GridDomain and rebuilds it on unpickle. ---
template<unsigned D>
struct SlimFilTraits<oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, D>>> {
    using Cell = oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, D>>;
    using Fil = oin::Filtration<Cell, oin_real>;
    using SlimValue = oin::CellWithValue<Cell, oin_real>;
    using FatValue = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>;
    using Geometry = oin::FrGeometry<oin_int, D>;
    using UidArg = unsigned __int128;

    static constexpr bool has_ctor = false;
    static constexpr bool has_boundary_matrix_rel = false;
    static std::string py_name() { return "_FreudenthalFiltration_" + std::to_string(D) + "D"; }
    static constexpr const char* cell_word = "simplex";
    static constexpr const char* cells_word = "simplices";
    static constexpr const char* value_word = "simplex_value_by_sorted_id";

    static FatValue fatten(const SlimValue& cv, const Geometry& g) { return fatten_simplex_from_fr<oin_int, D, oin_real>(cv, g); }
    static std::vector<FatValue> fatten_all_(const Fil& fil) { return fatten_all_fr<oin_int, D, oin_real>(fil); }
    static SlimValue slim(const FatValue& fc, const Geometry& g) { return slim_simplex_from_fr<oin_int, D, oin_real>(fc, g); }
    static typename Cell::Uid to_slim_uid(const Geometry& g, UidArg uid) { return fr_slim_uid_from_comb_uid<oin_int, D>(g, uid); }

    using StateTuple = std::tuple<decltype(Fil::negate_), std::vector<FatValue>, decltype(Fil::is_subfiltration_),
            decltype(Fil::uid_to_sorted_id), decltype(Fil::id_to_sorted_id_), decltype(Fil::sorted_id_to_id_),
            decltype(Fil::dim_first_), decltype(Fil::dim_last_), decltype(Fil::kind_), oin::GridDomain<oin_int, D>>;

    static StateTuple getstate(const Fil& fil)
    {
        auto fat = fatten_all_fr<oin_int, D, oin_real>(fil);
        return std::make_tuple(fil.negate_, std::move(fat), fil.is_subfiltration_,
                fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_,
                fil.kind_, fil.geometry_.domain);
    }
    static void setstate(Fil& fil, const StateTuple& t)
    {
        new (&fil) Fil();
        fil.negate_ = std::get<0>(t);
        fil.is_subfiltration_ = std::get<2>(t);
        fil.uid_to_sorted_id = std::get<3>(t);
        fil.id_to_sorted_id_ = std::get<4>(t);
        fil.sorted_id_to_id_ = std::get<5>(t);
        fil.dim_first_ = std::get<6>(t);
        fil.dim_last_ = std::get<7>(t);
        fil.kind_ = std::get<8>(t);
        // FrGeometry is not picklable; rebuild it from the stored GridDomain, then slim the fat
        // cells back against it and rebuild the flat uid index
        oin::FrGeometry<oin_int, D> frgeom(std::get<9>(t));
        fil.set_geometry(frgeom);
        const auto& fat = std::get<1>(t);
        typename Fil::CellVector slim_cells;
        slim_cells.reserve(fat.size());
        for (const auto& fc : fat) slim_cells.push_back(slim_simplex_from_fr<oin_int, D, oin_real>(fc, frgeom));
        fil.cells_ = std::move(slim_cells);
        fil.rebuild_uid_index_();
    }
};

// --- bit-packed VR/alpha: Simplex<Int, BitPacked<Int,Word>> materializes a fat Simplex. Geometry
// is a trivially-copyable PackedGeom{int bits}, so the state stores the bits int directly (no
// table-bearing geometry to rebuild) and OMITS uid_to_sorted_id (BitPacked uses the hash, which
// rebuild_uid_index_ regenerates -- the __int128 hash keys cannot cross the pickle boundary). The
// uid accessors translate the combinatorial uid to the packed Word via packed_word_uid_from_comb_uid. ---
template<class Word>
struct SlimFilTraits<oin::Simplex<oin_int, oin::BitPacked<oin_int, Word>>> {
    using Cell = oin::Simplex<oin_int, oin::BitPacked<oin_int, Word>>;
    using Fil = oin::Filtration<Cell, oin_real>;
    using SlimValue = oin::CellWithValue<Cell, oin_real>;
    using FatValue = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>;
    using Geometry = oin::PackedGeom;
    using UidArg = unsigned __int128;

    static constexpr bool has_ctor = false;
    static constexpr bool has_boundary_matrix_rel = false;
    static std::string py_name()
    {
        return std::string("_PackedSimplexFiltration_") + (sizeof(Word) <= 8 ? "64" : "128");
    }
    static constexpr const char* cell_word = "simplex";
    static constexpr const char* cells_word = "simplices";
    static constexpr const char* value_word = "simplex_value_by_sorted_id";

    static FatValue fatten(const SlimValue& cv, const Geometry& g) { return fatten_simplex_from_packed<oin_int, Word, oin_real>(cv, g); }
    static std::vector<FatValue> fatten_all_(const Fil& fil) { return fatten_all_packed<oin_int, Word, oin_real>(fil); }
    static SlimValue slim(const FatValue& fc, const Geometry& g) { return slim_simplex_from_packed<oin_int, Word, oin_real>(fc, g); }
    static typename Cell::Uid to_slim_uid(const Geometry& g, UidArg uid) { return packed_word_uid_from_comb_uid<oin_int, Word>(g, uid); }

    using StateTuple = std::tuple<decltype(Fil::negate_), std::vector<FatValue>, decltype(Fil::is_subfiltration_),
            decltype(Fil::id_to_sorted_id_), decltype(Fil::sorted_id_to_id_),
            decltype(Fil::dim_first_), decltype(Fil::dim_last_), decltype(Fil::kind_), int>;

    static StateTuple getstate(const Fil& fil)
    {
        auto fat = fatten_all_packed<oin_int, Word, oin_real>(fil);
        return std::make_tuple(fil.negate_, std::move(fat), fil.is_subfiltration_,
                fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_,
                fil.kind_, fil.geometry_.bits);
    }
    static void setstate(Fil& fil, const StateTuple& t)
    {
        new (&fil) Fil();
        fil.negate_ = std::get<0>(t);
        fil.is_subfiltration_ = std::get<2>(t);
        fil.id_to_sorted_id_ = std::get<3>(t);
        fil.sorted_id_to_id_ = std::get<4>(t);
        fil.dim_first_ = std::get<5>(t);
        fil.dim_last_ = std::get<6>(t);
        fil.kind_ = std::get<7>(t);
        oin::PackedGeom geom{std::get<8>(t)};
        fil.set_geometry(geom);
        const auto& fat = std::get<1>(t);
        typename Fil::CellVector slim_cells;
        slim_cells.reserve(fat.size());
        for (const auto& fc : fat) slim_cells.push_back(slim_simplex_from_packed<oin_int, Word, oin_real>(fc, geom));
        fil.cells_ = std::move(slim_cells);
        fil.rebuild_uid_index_();
    }
};

// Bind one slim filtration class (Cube / Freudenthal / packed) from the shared method set plus
// the per-policy SlimFilTraits hooks. Replaces BIND_CUBE/FR/PACKED_FILTRATION.
template<class Policy>
void register_slim_filtration(nb::module_& m)
{
    using T = SlimFilTraits<Policy>;
    using Fil = typename T::Fil;
    using FatValue = typename T::FatValue;
    using StateTuple = typename T::StateTuple;
    namespace nbp = oineus_python;

    auto cls = nb::class_<Fil>(m, T::py_name().c_str());

    if constexpr (T::has_ctor) {
        // ctor from fat cells: slim them, then recover the shared geometry from cell 0
        cls.def("__init__", [](Fil* p, const std::vector<FatValue>& fat_cells, bool negate, int n_threads) {
                typename T::Geometry dom;
                if (not fat_cells.empty()) dom = fat_cells[0].get_cell().global_domain();
                typename Fil::CellVector slim_cells;
                slim_cells.reserve(fat_cells.size());
                for (const auto& fc : fat_cells) slim_cells.push_back(T::slim(fc, dom));
                new (p) Fil(std::move(slim_cells), negate, n_threads);
                p->set_geometry(dom);
            }, nb::arg("cells"), nb::arg("negate") = false, nb::arg("n_threads") = 1);
    }

    cls.def("__len__", &Fil::size)
        .def("__iter__", [](const Fil& fil) -> nb::object {
                nb::object lst = nb::cast(T::fatten_all_(fil));
                return lst.attr("__iter__")();
            })
        .def("__getitem__", [](const Fil& fil, int i) {
                if (i < 0) i = fil.size() + i;
                return T::fatten(fil.get_cell(i), fil.geometry());
            })
        .def_prop_ro("negate", &Fil::negate)
        .def("infinity", &Fil::infinity, "filtration-order +inf")
        .def("neg_infinity", &Fil::neg_infinity, "filtration-order -inf")
        .def("fil_min", &Fil::fil_min, nb::arg("a"), nb::arg("b"), "filtration-order min")
        .def("fil_max", &Fil::fil_max, nb::arg("a"), nb::arg("b"), "filtration-order max")
        .def_prop_ro("max_dim", &Fil::max_dim, "maximal dimension of a cell in filtration")
        .def("cells", [](const Fil& fil) { return T::fatten_all_(fil); }, "copy of all cells in filtration order")
        .def(T::cells_word, [](const Fil& fil) { return T::fatten_all_(fil); }, "copy of all cells in filtration order")
        .def("size", &Fil::size, "number of cells in filtration")
        .def("size_in_dimension", &Fil::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim")
        .def("n_vertices", &Fil::n_vertices)
        .def(T::value_word, &Fil::value_by_sorted_id, nb::arg("sorted_id"))
        .def("cell_value_by_sorted_id", &Fil::value_by_sorted_id, nb::arg("sorted_id"))
        .def("id_by_sorted_id", &Fil::get_id_by_sorted_id, nb::arg("sorted_id"))
        .def("sorted_id_by_id", &Fil::get_sorted_id, nb::arg("id"))
        .def("cell", [](const Fil& fil, size_t i) { return T::fatten(fil.get_cell(i), fil.geometry()); }, nb::arg("i"))
        .def(T::cell_word, [](const Fil& fil, size_t i) { return T::fatten(fil.get_cell(i), fil.geometry()); }, nb::arg("i"))
        .def_prop_ro("dim_first", &Fil::dims_first)
        .def_prop_ro("dim_last", &Fil::dims_last)
        .def("sorting_permutation", &Fil::get_sorting_permutation)
        .def("inv_sorting_permutation", &Fil::get_inv_sorting_permutation)
        // uid accessors take the universal uid a Python cell carries (combinatorial for the
        // simplicial slim cells, the dense int for cubes); to_slim_uid re-keys it into this
        // filtration's internal uid
        .def("value_by_uid", [](const Fil& fil, typename T::UidArg uid) {
                return fil.value_by_uid(T::to_slim_uid(fil.geometry(), uid));
            }, nb::arg("uid"))
        .def("sorted_id_by_uid", [](const Fil& fil, typename T::UidArg uid) {
                return fil.get_sorted_id_by_uid(T::to_slim_uid(fil.geometry(), uid));
            }, nb::arg("uid"))
        .def("cell_by_uid", [](const Fil& fil, typename T::UidArg uid) {
                return T::fatten(fil.get_cell_by_uid(T::to_slim_uid(fil.geometry(), uid)), fil.geometry());
            }, nb::arg("uid"))
        .def("boundary_matrix", &Fil::boundary_matrix, nb::arg("n_threads") = 1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
        .def("boundary_matrix_in_dimension", &Fil::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads") = 1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
        .def("coboundary_matrix", &Fil::coboundary_matrix, nb::arg("n_threads") = 1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>());

    if constexpr (T::has_boundary_matrix_rel) {
        cls.def("boundary_matrix_rel", &Fil::boundary_matrix_rel);
    }

    cls.def("star_closure", &Fil::star_closure, nb::arg("seed_sorted_ids"), nb::arg("n_threads") = 1,
                nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                "Coface up-closure (union of stars) of the given cells (sorted_ids).")
        .def("is_up_closed", &Fil::is_up_closed, nb::arg("cells"), nb::arg("n_threads") = 1,
                nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                "True if the cells (sorted_ids) are closed under cofaces.")
        .def("without_cells", &Fil::without_cells, nb::arg("cells_to_remove"), nb::rv_policy::move,
                "Subfiltration with the given cells (sorted_ids) removed; survivors keep order.")
        .def("reset_ids_to_sorted_ids", &Fil::reset_ids_to_sorted_ids)
        .def("set_values", &Fil::set_values, nb::arg("new_values"), nb::arg("n_threads") = 1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__repr__", [](const Fil& fil) { std::stringstream ss; ss << fil; return ss.str(); })
        .def_prop_rw("kind", &Fil::kind, &Fil::set_kind,
            "FiltrationKind tag set by the constructor that built this filtration (or User for hand-built ones).")
        .def("__getstate__", [](const Fil& fil) -> StateTuple { return T::getstate(fil); })
        .def("__setstate__", [](Fil& fil, const StateTuple& t) { T::setstate(fil, t); });

    // Simplex-vertex extractors (parity with the fat Filtration), for the simplicial slim/packed
    // encodings only -- a slim/packed cell materializes its vertices from the shared geometry.
    // The differentiable Cech-Delaunay / weak-alpha paths use get_edges/get_triangles/
    // get_tetrahedra to recompute per-simplex values. Cubical cells are excluded (not simplices).
    if constexpr (is_simplex_cell<Policy>::value) {
        cls.def("get_vertices",   [](const Fil& self) { return extract_simplices_as_numpy(self, 0); })
           .def("get_edges",      [](const Fil& self) { return extract_simplices_as_numpy(self, 1); })
           .def("get_triangles",  [](const Fil& self) { return extract_simplices_as_numpy(self, 2); })
           .def("get_tetrahedra", [](const Fil& self) { return extract_simplices_as_numpy(self, 3); })
           // name matches the fat Filtration's binding for cross-encoding consistency
           .def("get_simplices_as_arr", [](const Fil& self, dim_type simplex_dim) { return extract_simplices_as_numpy(self, simplex_dim); },
                nb::arg("simplex_dim"));
    }

    // min_filtration over two slim/packed filtrations of this cell type. The C++ helper works
    // entirely in the internal (anchor,type)/packed/dense uid space (cell.get_uid() ->
    // fil_2.get_sorted_id_by_uid), so no combinatorial translation and no __int128 crosses the
    // Python boundary -- it takes/returns Filtration objects and size_t perms only. Folding it
    // here registers the overload for every slim cell type (cube / Freudenthal / bit-packed).
    m.def("_min_filtration", &oin::min_filtration<Policy, oin_real>,
          nb::arg("fil_1"), nb::arg("fil_2"),
          "return a filtration where each cell has minimal value from fil_1, fil_2");
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<Policy, oin_real>,
          nb::arg("fil_1"), nb::arg("fil_2"),
          "return a tuple (filtration, inds_1, inds_2) where each cell has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
}

void init_oineus_filtration(nb::module_& m)
{
    using Simplex = oin::Simplex<oin_int>;
    using Filtration = oin::Filtration<Simplex, oin_real>;

    using FiltrationStateTuple = std::tuple<decltype(Filtration::negate_),
                                            decltype(Filtration::cells_),
                                            decltype(Filtration::is_subfiltration_),
                                            decltype(Filtration::uid_to_sorted_id),
                                            decltype(Filtration::id_to_sorted_id_),
                                            decltype(Filtration::sorted_id_to_id_),
                                            decltype(Filtration::dim_first_),
                                            decltype(Filtration::dim_last_),
                                            decltype(Filtration::kind_)
                                           >;


    using ProdSimplex = oin::ProductCell<Simplex, Simplex>;
    using ProdFiltration = oin::Filtration<ProdSimplex, oin_real>;

    using ProdFiltrationStateTuple = std::tuple<decltype(ProdFiltration::negate_),
                                                decltype(ProdFiltration::cells_),
                                                decltype(ProdFiltration::is_subfiltration_),
                                                decltype(ProdFiltration::uid_to_sorted_id),
                                                decltype(ProdFiltration::id_to_sorted_id_),
                                                decltype(ProdFiltration::sorted_id_to_id_),
                                                decltype(ProdFiltration::dim_first_),
                                                decltype(ProdFiltration::dim_last_),
                                                decltype(ProdFiltration::kind_)
                                               >;

    using oin::VREdge;

    const std::string filtration_class_name = "_Filtration";
    const std::string prod_filtration_class_name = "_ProdFiltration";


    nb::class_<Filtration>(m, filtration_class_name.c_str())
            .def(nb::init<Filtration::CellVector, bool, int>(),
                    nb::arg("cells"),
                    nb::arg("negate") = false,
                    nb::arg("n_threads") = 1
                    )
            // this ctor accepts the output of Diode directly, list of (vertices, value)
            .def("__init__", [](Filtration* pfil, const std::vector<std::tuple<std::vector<unsigned>, oin_real>>& diode_simplices, int n_threads, bool duplicates_possible) {
                Filtration::CellVector oin_simplices;
                oin_simplices.reserve(diode_simplices.size());

                if (duplicates_possible) {
                    std::unordered_map<Simplex::Uid, size_t> uid_to_idx;
                    uid_to_idx.reserve(diode_simplices.size());

                    for(const auto& [vs_, val] : diode_simplices) {
                        Simplex::IdxVector vs;
                        vs.reserve(vs_.size());
                        for (unsigned v : vs_) { vs.push_back(v); }

                        Simplex sigma(vs);
                        auto uid = sigma.get_uid();
                        auto it = uid_to_idx.find(uid);

                        if (it == uid_to_idx.end()) {
                            uid_to_idx.emplace(uid, oin_simplices.size());
                            oin_simplices.emplace_back(std::move(sigma), val);
                        } else {
                            auto& existing = oin_simplices[it->second];
                            if (val < existing.value_) {
                                existing.value_ = val;
                            }
                        }
                    }
                } else {
                    for(const auto& [vs_, val] : diode_simplices) {
                        Simplex::IdxVector vs;
                        vs.reserve(vs_.size());
                        for (unsigned v : vs_) { vs.push_back(v); }
                        oin_simplices.emplace_back(Simplex(vs), val);
                    }
                }

                // Negation must stay false here; n_threads is the third ctor argument.
                new (pfil) Filtration(std::move(oin_simplices), false, n_threads);
            }, nb::arg("vertices_values"), nb::arg("n_threads") = 1, nb::arg("duplicates_possible") = false)
            .def("__len__", &Filtration::size)
            .def("__iter__", [](Filtration& fil) { return nb::make_iterator(nb::type<Filtration>(), "simplex_iterator", fil.begin(), fil.end()); }, nb::keep_alive<0, 1>())
            .def("__getitem__", [](Filtration& fil, int i) { if (i < 0) i = fil.size() + i; return fil.get_cell(i);})
            .def_prop_ro("negate", &Filtration::negate)
            .def("infinity", &Filtration::infinity, "filtration-order +inf: a value strictly later than every cell")
            .def("neg_infinity", &Filtration::neg_infinity, "filtration-order -inf: a value strictly earlier than every cell")
            .def("fil_min", &Filtration::fil_min, nb::arg("a"), nb::arg("b"), "filtration-order min: the value that enters earlier")
            .def("fil_max", &Filtration::fil_max, nb::arg("a"), nb::arg("b"), "filtration-order max: the value that enters later")
            .def_prop_ro("max_dim", &Filtration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &Filtration::cells_copy, "copy of all cells in filtration order")
            .def("simplices", &Filtration::cells_copy, "copy of all simplices (cells) in filtration order")
            .def("size", &Filtration::size, "number of cells in filtration")
            .def("size_in_dimension", &Filtration::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim")
            .def("n_vertices", &Filtration::n_vertices)
            .def("simplex_value_by_sorted_id", &Filtration::value_by_sorted_id, nb::arg("sorted_id"))
            .def("cell_value_by_sorted_id", &Filtration::value_by_sorted_id, nb::arg("sorted_id"))
            .def("id_by_sorted_id", &Filtration::get_id_by_sorted_id, nb::arg("sorted_id"))
            .def("sorted_id_by_id", &Filtration::get_sorted_id, nb::arg("id"))
            .def("cell", &Filtration::get_cell, nb::arg("i"))
            .def("simplex", &Filtration::get_cell, nb::arg("i"))
            .def_prop_ro("dim_first", &Filtration::dims_first)
            .def_prop_ro("dim_last", &Filtration::dims_last)
            .def("sorting_permutation", &Filtration::get_sorting_permutation)
            .def("inv_sorting_permutation", &Filtration::get_inv_sorting_permutation)
            .def("value_by_uid", &Filtration::value_by_uid, nb::arg("uid"))
            .def("sorted_id_by_uid", &Filtration::get_sorted_id_by_uid, nb::arg("uid"))
            .def("cell_by_uid", &Filtration::get_cell_by_uid, nb::arg("uid"))
            .def("boundary_matrix", &Filtration::boundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("boundary_matrix_in_dimension", &Filtration::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("coboundary_matrix", &Filtration::coboundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("boundary_matrix_rel", &Filtration::boundary_matrix_rel)
            .def("star_closure", &Filtration::star_closure, nb::arg("seed_sorted_ids"), nb::arg("n_threads")=1,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Coface up-closure (union of stars) of the given cells (sorted_ids); pass to "
                    "Decomposition.remove_simplices / Filtration.without_cells.")
            .def("is_up_closed", &Filtration::is_up_closed, nb::arg("cells"), nb::arg("n_threads")=1,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "True if the cells (sorted_ids) are closed under cofaces (removable as a filtration).")
            .def("without_cells", &Filtration::without_cells, nb::arg("cells_to_remove"), nb::rv_policy::move,
                    "Subfiltration with the given cells (sorted_ids) removed; survivors keep order.")
            .def("reset_ids_to_sorted_ids", &Filtration::reset_ids_to_sorted_ids)
            .def("set_values", &Filtration::set_values, nb::arg("new_values"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("subfiltration", [](Filtration& self, const
                std::function<bool(const Simplex&)>& py_pred) -> Filtration {
                auto pred = [&py_pred](const Filtration::Cell& c) -> bool { return py_pred(c.cell_);
                };
                // auto pred = [](const Filtration::Cell& cell) -> bool { return cell.dim() == 1; };
                auto result = self.subfiltration(pred);
                return result;
             }, nb::arg("predicate"), nb::rv_policy::copy)
             // .def("subfiltration", [](Filtration& self, const
                // std::function<bool(const Filtration::Cell&)>& py_pred) {
                // Filtration result = self.subfiltration(py_pred);
                // return result;
             // }, nb::arg("predicate"))
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def("get_vertices", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 0); })
            .def("get_edges", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 1); })
            .def("get_triangles", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 2); })
            .def("get_tetrahedra", [](Filtration& self) -> nb::ndarray<size_t, nb::numpy> { return extract_simplices_as_numpy(self, 3); })
            .def("get_simplices_as_arr", [](Filtration& self, dim_type simplex_dim) -> nb::ndarray<size_t, nb::numpy>
                { return extract_simplices_as_numpy(self, simplex_dim); }, nb::arg("simplex_dim"))
            .def("__repr__", [](const Filtration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            })
            .def_prop_rw("kind", &Filtration::kind, &Filtration::set_kind,
                "FiltrationKind tag set by the constructor that built this filtration "
                "(or User for hand-built ones).")
            .def("__getstate__", [](const Filtration& fil) -> FiltrationStateTuple {
                  return std::make_tuple(fil.negate_, fil.cells_, fil.is_subfiltration_,
                      fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_,
                      fil.kind_);
                })
            .def("__setstate__", [](Filtration& fil, const FiltrationStateTuple& t) {
                new (&fil) Filtration();
                fil.negate_ = std::get<0>(t);
                fil.cells_ = std::get<1>(t);
                fil.is_subfiltration_ = std::get<2>(t);
                fil.uid_to_sorted_id = std::get<3>(t);
                fil.id_to_sorted_id_ = std::get<4>(t);
                fil.sorted_id_to_id_ = std::get<5>(t);
                fil.dim_first_ = std::get<6>(t);
                fil.dim_last_ = std::get<7>(t);
                fil.kind_ = std::get<8>(t);
            })
    ;

    // Build a Filtration directly from diode's per-dimension array exporters
    // (fill_delaunay_arrays / fill_alpha_shapes_arrays), bypassing the slow
    // list-of-(vertices, value) path. verts_by_dim[d] is an (n_d, d+1) integer
    // array of vertex ids; vals_by_dim, if given, is a list of (n_d,) value
    // arrays (0 is used when omitted, e.g. for combinatorics-only Delaunay whose
    // consumer recomputes the values). No deduplication -- intended for
    // non-periodic input, which has no duplicate simplices.
    // Bit-packed variant of the diode-array builder (below): emits
    // Simplex<oin_int, BitPacked<oin_int, Word>> cells with a shared PackedGeom{bits},
    // bits chosen to hold the largest vertex id. Same validation + sort as the fat
    // builder; covers alpha / Cech-Delaunay / weak-alpha (all funnel through here).
    // Templated on Word so one definition serves both packed tiers; bound below as
    // _filtration_from_arrays_packed64 / _packed128.
    auto make_filtration_from_arrays_packed = [](auto word_tag) {
        using Word = typename decltype(word_tag)::type;
        using PackedCell = oin::Simplex<oin_int, oin::BitPacked<oin_int, Word>>;
        using PackedFil = oin::Filtration<PackedCell, oin_real>;
        return [](nb::list verts_by_dim, nb::object vals_by_dim, int n_threads,
                  int bits_arg, bool assume_sorted) -> PackedFil {
            using VertArr = nb::ndarray<oin_int,  nb::ndim<2>, nb::c_contig, nb::device::cpu, nb::ro>;
            using ValArr  = nb::ndarray<oin_real, nb::ndim<1>, nb::c_contig, nb::device::cpu, nb::ro>;

            const bool have_vals = !vals_by_dim.is_none();
            const size_t n_dims = nb::len(verts_by_dim);

            nb::list vals_list;
            if (have_vals) {
                vals_list = nb::cast<nb::list>(vals_by_dim);
                if (nb::len(vals_list) != n_dims)
                    throw std::runtime_error("_filtration_from_arrays_packed: vals_by_dim must have the same length as verts_by_dim");
            }

            std::vector<VertArr> vert_arrs;
            std::vector<ValArr> val_arrs;
            vert_arrs.reserve(n_dims);
            if (have_vals) val_arrs.reserve(n_dims);
            for (size_t d = 0; d < n_dims; ++d) {
                VertArr va = nb::cast<VertArr>(verts_by_dim[d]);
                if (static_cast<size_t>(va.shape(1)) != d + 1)
                    throw std::runtime_error("_filtration_from_arrays_packed: verts_by_dim[" + std::to_string(d)
                        + "] must have " + std::to_string(d + 1) + " columns");
                vert_arrs.push_back(va);
                if (have_vals) {
                    ValArr vala = nb::cast<ValArr>(vals_list[d]);
                    if (vala.shape(0) != va.shape(0))
                        throw std::runtime_error("_filtration_from_arrays_packed: vals_by_dim[" + std::to_string(d)
                            + "] length does not match verts_by_dim");
                    val_arrs.push_back(vala);
                }
            }

            PackedFil result;
            {
                oineus_python::SignalGuard guard;
                nb::gil_scoped_release release;

                // bits must hold the largest vertex id. The Python alpha/Delaunay
                // wrappers know n_points and pass bits directly, skipping this full
                // O(nnz) scan; bits_arg < 0 means "compute it here" (the safe
                // default for any other caller).
                int bits = bits_arg;
                if (bits < 0) {
                    oin_int max_id = 0;
                    for (const auto& va : vert_arrs) {
                        const oin_int* vp = va.data();
                        const size_t cnt = static_cast<size_t>(va.shape(0)) * static_cast<size_t>(va.shape(1));
                        for (size_t k = 0; k < cnt; ++k)
                            max_id = std::max(max_id, vp[k]);
                    }
                    bits = oin::packed_vertex_bits(static_cast<size_t>(max_id) + 1);
                }

                typename PackedFil::CellVector cells;
                size_t total = 0;
                for (const auto& va : vert_arrs)
                    total += va.shape(0);
                cells.reserve(total);

                // Zero-copy view over one numpy row so the sorted path packs
                // straight from the array (no per-simplex heap temp); begin/end
                // satisfy BitPacked's is_sorted debug assert.
                struct RowView {
                    const oin_int* p; size_t n;
                    size_t size() const { return n; }
                    oin_int operator[](size_t i) const { return p[i]; }
                    const oin_int* begin() const { return p; }
                    const oin_int* end() const { return p + n; }
                };
                std::vector<oin_int> buf;   // reused across simplices on the unsorted path
                for (size_t d = 0; d < vert_arrs.size(); ++d) {
                    const oin_int* vp = vert_arrs[d].data();
                    const oin_real* valp = have_vals ? val_arrs[d].data() : nullptr;
                    const size_t n = vert_arrs[d].shape(0);
                    const size_t w = vert_arrs[d].shape(1);
                    for (size_t i = 0; i < n; ++i) {
                        const oin_int* row = vp + i * w;
                        const oin_real val = valp ? valp[i] : oin_real(0);
                        if (assume_sorted) {
                            cells.emplace_back(PackedCell(oin::BitPacked<oin_int, Word>(RowView{row, w}, bits)), val);
                        } else {
                            buf.assign(row, row + w);
                            if (w > 1)
                                std::sort(buf.begin(), buf.end());
                            cells.emplace_back(PackedCell(oin::BitPacked<oin_int, Word>(buf, bits)), val);
                        }
                    }
                }

                result = PackedFil(std::move(cells), false, std::max(1, n_threads));
                result.set_geometry(oin::PackedGeom{bits});
            }
            return result;
        };
    };

    {
        struct W64 { using type = std::uint64_t; };
        struct W128 { using type = unsigned __int128; };
        m.def("_filtration_from_arrays_packed64", make_filtration_from_arrays_packed(W64{}),
            nb::arg("verts_by_dim"), nb::arg("vals_by_dim") = nb::none(), nb::arg("n_threads") = 1,
            nb::arg("bits") = -1, nb::arg("assume_sorted") = false,
            "Bit-packed (uint64) variant of _filtration_from_arrays (alpha/Delaunay). "
            "bits>=0 skips the max-id scan; assume_sorted=True skips the per-simplex vertex sort.");
        m.def("_filtration_from_arrays_packed128", make_filtration_from_arrays_packed(W128{}),
            nb::arg("verts_by_dim"), nb::arg("vals_by_dim") = nb::none(), nb::arg("n_threads") = 1,
            nb::arg("bits") = -1, nb::arg("assume_sorted") = false,
            "Bit-packed (unsigned __int128) variant of _filtration_from_arrays (alpha/Delaunay). "
            "bits>=0 skips the max-id scan; assume_sorted=True skips the per-simplex vertex sort.");
    }

    m.def("_filtration_from_arrays",
        [](nb::list verts_by_dim, nb::object vals_by_dim, int n_threads) -> Filtration {
            using VertArr = nb::ndarray<oin_int,  nb::ndim<2>, nb::c_contig, nb::device::cpu, nb::ro>;
            using ValArr  = nb::ndarray<oin_real, nb::ndim<1>, nb::c_contig, nb::device::cpu, nb::ro>;

            const bool have_vals = !vals_by_dim.is_none();
            const size_t n_dims = nb::len(verts_by_dim);

            nb::list vals_list;
            if (have_vals) {
                vals_list = nb::cast<nb::list>(vals_by_dim);
                if (nb::len(vals_list) != n_dims)
                    throw std::runtime_error("_filtration_from_arrays: vals_by_dim must have the same length as verts_by_dim");
            }

            // Capture the numpy views (keeps them alive) and validate shapes while
            // the GIL is held; the heavy build + sort below runs with it released.
            std::vector<VertArr> vert_arrs;
            std::vector<ValArr> val_arrs;
            vert_arrs.reserve(n_dims);
            if (have_vals) val_arrs.reserve(n_dims);

            for (size_t d = 0; d < n_dims; ++d) {
                VertArr va = nb::cast<VertArr>(verts_by_dim[d]);
                if (static_cast<size_t>(va.shape(1)) != d + 1)
                    throw std::runtime_error("_filtration_from_arrays: verts_by_dim[" + std::to_string(d)
                        + "] must have " + std::to_string(d + 1) + " columns");
                vert_arrs.push_back(va);
                if (have_vals) {
                    ValArr vala = nb::cast<ValArr>(vals_list[d]);
                    if (vala.shape(0) != va.shape(0))
                        throw std::runtime_error("_filtration_from_arrays: vals_by_dim[" + std::to_string(d)
                            + "] length does not match verts_by_dim");
                    val_arrs.push_back(vala);
                }
            }

            // SignalGuard + manual GIL release (matches oineus_functions.cpp): the
            // build and the Filtration ctor's parallel sort run with the GIL
            // released, and the ctor's interrupt-polling site stays responsive to
            // Ctrl-C. SignalGuard is declared first so it is destroyed last, after
            // the GIL has been reacquired by release's destructor.
            Filtration result;
            {
                oineus_python::SignalGuard guard;
                nb::gil_scoped_release release;

                Filtration::CellVector cells;
                size_t total = 0;
                for (const auto& va : vert_arrs)
                    total += va.shape(0);
                cells.reserve(total);

                for (size_t d = 0; d < vert_arrs.size(); ++d) {
                    const oin_int* vp = vert_arrs[d].data();
                    const oin_real* valp = have_vals ? val_arrs[d].data() : nullptr;
                    const size_t n = vert_arrs[d].shape(0);
                    const size_t w = vert_arrs[d].shape(1);
                    for (size_t i = 0; i < n; ++i) {
                        Simplex::IdxVector vs;
                        vs.reserve(w);
                        for (size_t j = 0; j < w; ++j)
                            vs.push_back(vp[i * w + j]);
                        if (w > 1)
                            std::sort(vs.begin(), vs.end());
                        // move the sorted vector into the Simplex's storage instead of
                        // letting Simplex(const IdxVector&) copy and re-sort it
                        cells.emplace_back(Simplex(oin::presorted_t{}, std::move(vs)), valp ? valp[i] : oin_real(0));
                    }
                }

                result = Filtration(std::move(cells), false, std::max(1, n_threads));
            }
            return result;
        },
        nb::arg("verts_by_dim"), nb::arg("vals_by_dim") = nb::none(), nb::arg("n_threads") = 1,
        "Build a Filtration from diode's per-dimension array exporters. "
        "verts_by_dim[d] is an (n_d, d+1) integer array of vertex ids; optional "
        "vals_by_dim[d] is an (n_d,) array of filtration values (0 when omitted).");

    nb::class_<ProdFiltration>(m, prod_filtration_class_name.c_str())
            .def(nb::init<ProdFiltration::CellVector, bool, int>(),
                    nb::arg("cells"),
                    nb::arg("negate") = false,
                    nb::arg("n_threads") = 1
                    )
            .def("__len__", &ProdFiltration::size)
            .def("__iter__", [](ProdFiltration& fil) { return nb::make_iterator(nb::type<ProdFiltration>(), "simplex_simplex_iterator", fil.begin(), fil.end()); }, nb::keep_alive<0,
            1>())
            .def("__getitem__", [](ProdFiltration& fil, size_t i) { return fil.get_cell(i);})
            .def_prop_ro("negate", &ProdFiltration::negate)
            .def("infinity", &ProdFiltration::infinity, "filtration-order +inf: a value strictly later than every cell")
            .def("neg_infinity", &ProdFiltration::neg_infinity, "filtration-order -inf: a value strictly earlier than every cell")
            .def("fil_min", &ProdFiltration::fil_min, nb::arg("a"), nb::arg("b"), "filtration-order min: the value that enters earlier")
            .def("fil_max", &ProdFiltration::fil_max, nb::arg("a"), nb::arg("b"), "filtration-order max: the value that enters later")
            .def_prop_ro("max_dim", &ProdFiltration::max_dim, "maximal dimension of a cell in filtration")
            .def("cells", &ProdFiltration::cells_copy, "copy of all cells in filtration order")
            .def("size", &ProdFiltration::size, "number of cells in filtration")
            .def_prop_ro("dim_first", &ProdFiltration::dims_first)
            .def_prop_ro("dim_last", &ProdFiltration::dims_last)
            .def("size_in_dimension", &ProdFiltration::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim")
            .def("cell_value_by_sorted_id", &ProdFiltration::value_by_sorted_id, nb::arg("sorted_id"))
            .def("get_id_by_sorted_id", &ProdFiltration::get_id_by_sorted_id, nb::arg("sorted_id"))
            .def("get_sorted_id_by_id", &ProdFiltration::get_sorted_id, nb::arg("id"))
            .def("get_cell", &ProdFiltration::get_cell, nb::arg("i"))
            .def("get_sorting_permutation", &ProdFiltration::get_sorting_permutation)
            .def("get_inv_sorting_permutation", &ProdFiltration::get_inv_sorting_permutation)
            // unprefixed aliases matching every other filtration binding (Simplex / Cube /
            // Freudenthal / Packed), so a facade can treat all filtration types uniformly;
            // the get_-prefixed forms above are kept for backward compatibility
            .def("id_by_sorted_id", &ProdFiltration::get_id_by_sorted_id, nb::arg("sorted_id"))
            .def("sorted_id_by_id", &ProdFiltration::get_sorted_id, nb::arg("id"))
            .def("cell", &ProdFiltration::get_cell, nb::arg("i"))
            .def("sorting_permutation", &ProdFiltration::get_sorting_permutation)
            .def("inv_sorting_permutation", &ProdFiltration::get_inv_sorting_permutation)
            .def("boundary_matrix", &ProdFiltration::boundary_matrix, nb::arg("n_threads"), nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("boundary_matrix_in_dimension", &ProdFiltration::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads"), nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("coboundary_matrix", &ProdFiltration::coboundary_matrix, nb::arg("n_threads"), nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("star_closure", &ProdFiltration::star_closure, nb::arg("seed_sorted_ids"), nb::arg("n_threads")=1,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "Coface up-closure (union of stars) of the given cells (sorted_ids).")
            .def("is_up_closed", &ProdFiltration::is_up_closed, nb::arg("cells"), nb::arg("n_threads")=1,
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(),
                    "True if the cells (sorted_ids) are closed under cofaces.")
            .def("without_cells", &ProdFiltration::without_cells, nb::arg("cells_to_remove"), nb::rv_policy::move,
                    "Subfiltration with the given cells (sorted_ids) removed; survivors keep order.")
            .def("reset_ids_to_sorted_ids", &ProdFiltration::reset_ids_to_sorted_ids)
            .def("set_values", &ProdFiltration::set_values, nb::arg("new_values"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>())
            .def("__repr__", [](const ProdFiltration& fil) {
              std::stringstream ss;
              ss << fil;
              return ss.str();
            })
            .def(nb::self == nb::self)
            .def(nb::self != nb::self)
            .def_prop_rw("kind", &ProdFiltration::kind, &ProdFiltration::set_kind,
                "FiltrationKind tag set by the constructor that built this filtration "
                "(or User for hand-built ones).")
            .def("__getstate__", [](const ProdFiltration& fil) -> ProdFiltrationStateTuple {
                  return std::make_tuple(fil.negate_, fil.cells_, fil.is_subfiltration_,
                      fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_,
                      fil.kind_);
                })
            .def("__setstate__", [](ProdFiltration& fil, const ProdFiltrationStateTuple& t) {
                new (&fil) ProdFiltration();
                fil.negate_ = std::get<0>(t);
                fil.cells_ = std::get<1>(t);
                fil.is_subfiltration_ = std::get<2>(t);
                fil.uid_to_sorted_id = std::get<3>(t);
                fil.id_to_sorted_id_ = std::get<4>(t);
                fil.sorted_id_to_id_ = std::get<5>(t);
                fil.dim_first_ = std::get<6>(t);
                fil.dim_last_ = std::get<7>(t);
                fil.kind_ = std::get<8>(t);
            })
    ;

    // ============ Slim filtration bindings (Cube / Freudenthal / bit-packed) ============
    // register_slim_filtration<Policy> (above) binds the shared method set once; for_each_type
    // instantiates it for every slim cell type, and SlimFilTraits<Policy> supplies the per-policy
    // hooks (fat<->slim conversion, uid translation, pickle state, display words, the cube-only
    // ctor + boundary_matrix_rel). This replaces the former BIND_CUBE/FR/PACKED_FILTRATION macros.
    oineus_python::for_each_type(
        oineus_python::TypeList<
            oin::Cube<oin_int, 1>, oin::Cube<oin_int, 2>, oin::Cube<oin_int, 3>,
            oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 1>>,
            oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 2>>,
            oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, 3>>,
            oin::Simplex<oin_int, oin::BitPacked<oin_int, std::uint64_t>>,
            oin::Simplex<oin_int, oin::BitPacked<oin_int, unsigned __int128>>
        >{},
        [&m]<class Policy>() { register_slim_filtration<Policy>(m); });

    m.def("_mapping_cylinder",
          [](const Filtration& fil_domain, const Filtration& fil_codomain,
             const Simplex& v_domain, const Simplex& v_codomain,
             oin_real v_domain_value, oin_real v_codomain_value) {
              return oin::build_mapping_cylinder<Simplex, oin_real>(
                  fil_domain, fil_codomain, v_domain, v_codomain,
                  v_domain_value, v_codomain_value);
          },
          nb::arg("fil_domain"), nb::arg("fil_codomain"),
          nb::arg("v_domain"), nb::arg("v_codomain"),
          nb::arg("v_domain_value"), nb::arg("v_codomain_value"),
          "return mapping cylinder filtration of inclusion fil_domain -> fil_codomain. "
          "v_domain_value/v_codomain_value are the filtration values assigned to the "
          "auxiliary vertices; pass fil.neg_infinity() to leave the cylinder's persistent "
          "homology equivalent to the inclusion's.");
    m.def("_mapping_cylinder_with_indices",
          [](const Filtration& fil_domain, const Filtration& fil_codomain,
             const Simplex& v_domain, const Simplex& v_codomain,
             oin_real v_domain_value, oin_real v_codomain_value) {
              return oin::build_mapping_cylinder_with_indices<Simplex, oin_real>(
                  fil_domain, fil_codomain, v_domain, v_codomain,
                  v_domain_value, v_codomain_value);
          },
          nb::arg("fil_domain"), nb::arg("fil_codomain"),
          nb::arg("v_domain"), nb::arg("v_codomain"),
          nb::arg("v_domain_value"), nb::arg("v_codomain_value"),
          "return (mapping cylinder filtration, indices of critical values). See _mapping_cylinder.");
    m.def("_multiply_filtration",
          [](const Filtration& fil, const Simplex& sigma, oin_real sigma_value) {
              return oin::multiply_filtration<Simplex, oin_real>(fil, sigma, sigma_value);
          },
          nb::arg("fil"), nb::arg("sigma"), nb::arg("sigma_value"),
          "return a filtration with each simplex in fil multiplied by simplex sigma. "
          "Each product cell receives value fil_max(cell.value, sigma_value).");

    m.def("_min_filtration", &oin::min_filtration<Simplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a filtration where each simplex has minimal value from fil_1, fil_2");
    m.def("_min_filtration", &oin::min_filtration<ProdSimplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a filtration where each cell has minimal value from fil_1, fil_2");

    // helper for differentiable filtration
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<Simplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
    m.def("_min_filtration_with_indices", &oin::min_filtration_with_indices<ProdSimplex, oin_real>, nb::arg("fil_1"), nb::arg("fil_2"), "return a tuple (filtration, inds_1, inds_2) where each simplex has minimal value from fil_1, fil_2 and inds_1, inds_2 are its indices in fil_1, fil_2");
    // The slim/packed overloads of _min_filtration[_with_indices] are registered inside
    // register_slim_filtration<Policy> (folded over every slim cell type).

}
