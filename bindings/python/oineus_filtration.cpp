#include <functional>
#include "oineus_persistence_bindings.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/unordered_map.h>


nb::ndarray<size_t, nb::numpy> extract_simplices_as_numpy(const oin::Filtration<oin::Simplex<oin_int>, oin_real>& fil, dim_type simplex_dim)
{
    using VertexIndex = size_t;
    const dim_type simplex_size = simplex_dim + 1;
    const VertexIndex n_simplices = fil.size_in_dimension(simplex_dim);

    auto* simplices = new VertexIndex[simplex_size * n_simplices];

    for(auto simplex_idx = fil.dim_first(simplex_dim); simplex_idx <= fil.dim_last(simplex_dim); ++simplex_idx) {
        size_t array_idx = simplex_idx - fil.dim_first(simplex_dim);
        assert(fil.get_cell(simplex_idx).get_vertices().size() == simplex_size);
        for(size_t v_idx = 0; v_idx < simplex_size; ++v_idx) {
            simplices[simplex_size * array_idx + v_idx] = fil.get_cell(simplex_idx).get_vertices()[v_idx];
        }
    }

    nb::capsule free_when_done(simplices, [](void* p) noexcept {
     auto* pp = reinterpret_cast<VertexIndex*>(p);
     delete[] pp;
   });

    return nb::ndarray<VertexIndex, nb::numpy>(simplices, {n_simplices, simplex_size}, free_when_done);
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

    const std::string filtration_class_name = "Filtration";
    const std::string prod_filtration_class_name = "ProdFiltration";


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
        return [](nb::list verts_by_dim, nb::object vals_by_dim, int n_threads) -> PackedFil {
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

                // bits must hold the largest vertex id seen across all arrays
                oin_int max_id = 0;
                for (const auto& va : vert_arrs) {
                    const oin_int* vp = va.data();
                    const size_t cnt = static_cast<size_t>(va.shape(0)) * static_cast<size_t>(va.shape(1));
                    for (size_t k = 0; k < cnt; ++k)
                        max_id = std::max(max_id, vp[k]);
                }
                const int bits = oin::packed_vertex_bits(static_cast<size_t>(max_id) + 1);

                typename PackedFil::CellVector cells;
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
                        std::vector<oin_int> vs;
                        vs.reserve(w);
                        for (size_t j = 0; j < w; ++j)
                            vs.push_back(vp[i * w + j]);
                        if (w > 1)
                            std::sort(vs.begin(), vs.end());
                        cells.emplace_back(PackedCell(oin::BitPacked<oin_int, Word>(vs, bits)),
                                           valp ? valp[i] : oin_real(0));
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
            "Bit-packed (uint64) variant of _filtration_from_arrays (alpha/Delaunay).");
        m.def("_filtration_from_arrays_packed128", make_filtration_from_arrays_packed(W128{}),
            nb::arg("verts_by_dim"), nb::arg("vals_by_dim") = nb::none(), nb::arg("n_threads") = 1,
            "Bit-packed (unsigned __int128) variant of _filtration_from_arrays (alpha/Delaunay).");
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

    // ============ CubeFiltration bindings ============
    // The filtration stores slim cubes (oin::Cube) + one shared GridDomain;
    // every cube-returning accessor materializes the fat Python cube
    // (CellWithValue<FatCube>) via fatten_cube + the filtration's geometry(), and
    // the constructor/unpickle slim them back (slim_cube), recovering the shared
    // domain from the cells.
    #define BIND_CUBE_FILTRATION(DIM) \
        using CubeFiltration_##DIM##D = oin::Filtration<oin::Cube<oin_int, DIM>, oin_real>; \
        using FatCubeValue_##DIM##D = oin::CellWithValue<oin::FatCube<oin_int, DIM>, oin_real>; \
        using CubeFiltration_##DIM##DStateTuple = std::tuple<decltype(CubeFiltration_##DIM##D::negate_), \
                                                std::vector<FatCubeValue_##DIM##D>, \
                                                decltype(CubeFiltration_##DIM##D::is_subfiltration_), \
                                                decltype(CubeFiltration_##DIM##D::uid_to_sorted_id), \
                                                decltype(CubeFiltration_##DIM##D::id_to_sorted_id_), \
                                                decltype(CubeFiltration_##DIM##D::sorted_id_to_id_), \
                                                decltype(CubeFiltration_##DIM##D::dim_first_), \
                                                decltype(CubeFiltration_##DIM##D::dim_last_), \
                                                decltype(CubeFiltration_##DIM##D::kind_), \
                                                decltype(CubeFiltration_##DIM##D::geometry_) \
                                               >; \
        nb::class_<CubeFiltration_##DIM##D>(m, "CubeFiltration_" #DIM "D") \
            .def("__init__", [](CubeFiltration_##DIM##D* p, const std::vector<FatCubeValue_##DIM##D>& fat_cells, bool negate, int n_threads) { \
                    oin::GridDomain<oin_int, DIM> dom; \
                    if (not fat_cells.empty()) dom = fat_cells[0].get_cell().global_domain(); \
                    typename CubeFiltration_##DIM##D::CellVector slim; \
                    slim.reserve(fat_cells.size()); \
                    for (const auto& fc : fat_cells) slim.push_back(slim_cube<oin_int, DIM, oin_real>(fc)); \
                    new (p) CubeFiltration_##DIM##D(std::move(slim), negate, n_threads); \
                    p->set_geometry(dom); \
                }, nb::arg("cells"), nb::arg("negate") = false, nb::arg("n_threads") = 1) \
            .def("__len__", &CubeFiltration_##DIM##D::size) \
            .def("__iter__", [](const CubeFiltration_##DIM##D& fil) -> nb::object { \
                    nb::object lst = nb::cast(fatten_all<oin_int, DIM, oin_real>(fil)); \
                    return lst.attr("__iter__")(); \
                }) \
            .def("__getitem__", [](const CubeFiltration_##DIM##D& fil, int i) { \
                    if (i < 0) i = fil.size() + i; \
                    return fatten_cube<oin_int, DIM, oin_real>(fil.get_cell(i), fil.geometry()); \
                }) \
            .def_prop_ro("negate", &CubeFiltration_##DIM##D::negate) \
            .def("infinity", &CubeFiltration_##DIM##D::infinity, "filtration-order +inf") \
            .def("neg_infinity", &CubeFiltration_##DIM##D::neg_infinity, "filtration-order -inf") \
            .def("fil_min", &CubeFiltration_##DIM##D::fil_min, nb::arg("a"), nb::arg("b"), "filtration-order min") \
            .def("fil_max", &CubeFiltration_##DIM##D::fil_max, nb::arg("a"), nb::arg("b"), "filtration-order max") \
            .def_prop_ro("max_dim", &CubeFiltration_##DIM##D::max_dim, "maximal dimension of a cell in filtration") \
            .def("cells", [](const CubeFiltration_##DIM##D& fil) { return fatten_all<oin_int, DIM, oin_real>(fil); }, \
                    "copy of all cells in filtration order") \
            .def("cubes", [](const CubeFiltration_##DIM##D& fil) { return fatten_all<oin_int, DIM, oin_real>(fil); }, \
                    "copy of all cells in filtration order") \
            .def("size", &CubeFiltration_##DIM##D::size, "number of cells in filtration") \
            .def("size_in_dimension", &CubeFiltration_##DIM##D::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim") \
            .def("n_vertices", &CubeFiltration_##DIM##D::n_vertices) \
            .def("cube_value_by_sorted_id", &CubeFiltration_##DIM##D::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("cell_value_by_sorted_id", &CubeFiltration_##DIM##D::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("id_by_sorted_id", &CubeFiltration_##DIM##D::get_id_by_sorted_id, nb::arg("sorted_id")) \
            .def("sorted_id_by_id", &CubeFiltration_##DIM##D::get_sorted_id, nb::arg("id")) \
            .def("cell", [](const CubeFiltration_##DIM##D& fil, size_t i) { \
                    return fatten_cube<oin_int, DIM, oin_real>(fil.get_cell(i), fil.geometry()); \
                }, nb::arg("i")) \
            .def("cube", [](const CubeFiltration_##DIM##D& fil, size_t i) { \
                    return fatten_cube<oin_int, DIM, oin_real>(fil.get_cell(i), fil.geometry()); \
                }, nb::arg("i")) \
            .def_prop_ro("dim_first", &CubeFiltration_##DIM##D::dims_first) \
            .def_prop_ro("dim_last", &CubeFiltration_##DIM##D::dims_last) \
            .def("sorting_permutation", &CubeFiltration_##DIM##D::get_sorting_permutation) \
            .def("inv_sorting_permutation", &CubeFiltration_##DIM##D::get_inv_sorting_permutation) \
            .def("value_by_uid", &CubeFiltration_##DIM##D::value_by_uid, nb::arg("uid")) \
            .def("sorted_id_by_uid", &CubeFiltration_##DIM##D::get_sorted_id_by_uid, nb::arg("uid")) \
            .def("cell_by_uid", [](const CubeFiltration_##DIM##D& fil, oin_int uid) { \
                    return fatten_cube<oin_int, DIM, oin_real>(fil.get_cell_by_uid(uid), fil.geometry()); \
                }, nb::arg("uid")) \
            .def("boundary_matrix", &CubeFiltration_##DIM##D::boundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("boundary_matrix_in_dimension", &CubeFiltration_##DIM##D::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("coboundary_matrix", &CubeFiltration_##DIM##D::coboundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("boundary_matrix_rel", &CubeFiltration_##DIM##D::boundary_matrix_rel) \
            .def("star_closure", &CubeFiltration_##DIM##D::star_closure, nb::arg("seed_sorted_ids"), nb::arg("n_threads")=1, \
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(), \
                    "Coface up-closure (union of stars) of the given cells (sorted_ids).") \
            .def("is_up_closed", &CubeFiltration_##DIM##D::is_up_closed, nb::arg("cells"), nb::arg("n_threads")=1, \
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(), \
                    "True if the cells (sorted_ids) are closed under cofaces.") \
            .def("without_cells", &CubeFiltration_##DIM##D::without_cells, nb::arg("cells_to_remove"), nb::rv_policy::move, \
                    "Subfiltration with the given cells (sorted_ids) removed; survivors keep order.") \
            .def("reset_ids_to_sorted_ids", &CubeFiltration_##DIM##D::reset_ids_to_sorted_ids) \
            .def("set_values", &CubeFiltration_##DIM##D::set_values, nb::arg("new_values"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def("__repr__", [](const CubeFiltration_##DIM##D& fil) { \
                    std::stringstream ss; \
                    ss << fil; \
                    return ss.str(); \
                }) \
            .def_prop_rw("kind", &CubeFiltration_##DIM##D::kind, &CubeFiltration_##DIM##D::set_kind, \
                "FiltrationKind tag set by the constructor that built this filtration " \
                "(or User for hand-built ones).") \
            .def("__getstate__", [](const CubeFiltration_##DIM##D& fil) -> CubeFiltration_##DIM##DStateTuple { \
                      auto fat = fatten_all<oin_int, DIM, oin_real>(fil); \
                      return std::make_tuple(fil.negate_, std::move(fat), fil.is_subfiltration_, \
                          fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_, \
                          fil.kind_, fil.geometry_); \
                    }) \
            .def("__setstate__", [](CubeFiltration_##DIM##D& fil, const CubeFiltration_##DIM##DStateTuple& t) { \
                new (&fil) CubeFiltration_##DIM##D(); \
                fil.negate_ = std::get<0>(t); \
                const auto& fat = std::get<1>(t); \
                typename CubeFiltration_##DIM##D::CellVector slim; \
                slim.reserve(fat.size()); \
                for (const auto& fc : fat) slim.push_back(slim_cube<oin_int, DIM, oin_real>(fc)); \
                fil.cells_ = std::move(slim); \
                fil.is_subfiltration_ = std::get<2>(t); \
                fil.uid_to_sorted_id = std::get<3>(t); \
                fil.id_to_sorted_id_ = std::get<4>(t); \
                fil.sorted_id_to_id_ = std::get<5>(t); \
                fil.dim_first_ = std::get<6>(t); \
                fil.dim_last_ = std::get<7>(t); \
                fil.kind_ = std::get<8>(t); \
                fil.geometry_ = std::get<9>(t); \
                /* cube uses the flat uid index, not the (empty) hash above; */ \
                /* rebuild it from the restored cells (setstate bypasses the ctor) */ \
                fil.rebuild_uid_index_(); \
            }) \


    BIND_CUBE_FILTRATION(1);
    BIND_CUBE_FILTRATION(2);
    BIND_CUBE_FILTRATION(3);

    #undef BIND_CUBE_FILTRATION

    // ============ FreudenthalFiltration (slim) bindings ============
    // The slim Freudenthal filtration stores compact (anchor,type) Simplex cells +
    // one shared FrGeometry; every cell-returning accessor materializes the fat
    // Python simplex (CellWithValue<Simplex<oin_int>> == the "Simplex" class) via
    // fatten_simplex_from_fr + the filtration's geometry(). It is factory-produced
    // (Grid.freudenthal_filtration_slim / oineus.freudenthal_filtration), so there is
    // no public ctor-from-cells: a fat simplex carries no grid domain, so the shared
    // FrGeometry cannot be recovered from cells alone. Pickle stores the GridDomain
    // (a bound type, unlike FrGeometry) and rebuilds the geometry + slims the cells
    // back on restore.
    #define BIND_FR_FILTRATION(DIM) \
        using FrFiltration_##DIM##D = oin::Filtration<oin::Simplex<oin_int, oin::FreudenthalAnchorType<oin_int, DIM>>, oin_real>; \
        using FatSimplexValue_##DIM##D = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>; \
        using FrFiltration_##DIM##DStateTuple = std::tuple<decltype(FrFiltration_##DIM##D::negate_), \
                                                std::vector<FatSimplexValue_##DIM##D>, \
                                                decltype(FrFiltration_##DIM##D::is_subfiltration_), \
                                                decltype(FrFiltration_##DIM##D::uid_to_sorted_id), \
                                                decltype(FrFiltration_##DIM##D::id_to_sorted_id_), \
                                                decltype(FrFiltration_##DIM##D::sorted_id_to_id_), \
                                                decltype(FrFiltration_##DIM##D::dim_first_), \
                                                decltype(FrFiltration_##DIM##D::dim_last_), \
                                                decltype(FrFiltration_##DIM##D::kind_), \
                                                oin::GridDomain<oin_int, DIM> \
                                               >; \
        nb::class_<FrFiltration_##DIM##D>(m, "FreudenthalFiltration_" #DIM "D") \
            .def("__len__", &FrFiltration_##DIM##D::size) \
            .def("__iter__", [](const FrFiltration_##DIM##D& fil) -> nb::object { \
                    nb::object lst = nb::cast(fatten_all_fr<oin_int, DIM, oin_real>(fil)); \
                    return lst.attr("__iter__")(); \
                }) \
            .def("__getitem__", [](const FrFiltration_##DIM##D& fil, int i) { \
                    if (i < 0) i = fil.size() + i; \
                    return fatten_simplex_from_fr<oin_int, DIM, oin_real>(fil.get_cell(i), fil.geometry()); \
                }) \
            .def_prop_ro("negate", &FrFiltration_##DIM##D::negate) \
            .def("infinity", &FrFiltration_##DIM##D::infinity, "filtration-order +inf") \
            .def("neg_infinity", &FrFiltration_##DIM##D::neg_infinity, "filtration-order -inf") \
            .def("fil_min", &FrFiltration_##DIM##D::fil_min, nb::arg("a"), nb::arg("b"), "filtration-order min") \
            .def("fil_max", &FrFiltration_##DIM##D::fil_max, nb::arg("a"), nb::arg("b"), "filtration-order max") \
            .def_prop_ro("max_dim", &FrFiltration_##DIM##D::max_dim, "maximal dimension of a cell in filtration") \
            .def("cells", [](const FrFiltration_##DIM##D& fil) { return fatten_all_fr<oin_int, DIM, oin_real>(fil); }, \
                    "copy of all cells in filtration order") \
            .def("simplices", [](const FrFiltration_##DIM##D& fil) { return fatten_all_fr<oin_int, DIM, oin_real>(fil); }, \
                    "copy of all cells in filtration order") \
            .def("size", &FrFiltration_##DIM##D::size, "number of cells in filtration") \
            .def("size_in_dimension", &FrFiltration_##DIM##D::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim") \
            .def("n_vertices", &FrFiltration_##DIM##D::n_vertices) \
            .def("cell_value_by_sorted_id", &FrFiltration_##DIM##D::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("simplex_value_by_sorted_id", &FrFiltration_##DIM##D::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("id_by_sorted_id", &FrFiltration_##DIM##D::get_id_by_sorted_id, nb::arg("sorted_id")) \
            .def("sorted_id_by_id", &FrFiltration_##DIM##D::get_sorted_id, nb::arg("id")) \
            .def("cell", [](const FrFiltration_##DIM##D& fil, size_t i) { \
                    return fatten_simplex_from_fr<oin_int, DIM, oin_real>(fil.get_cell(i), fil.geometry()); \
                }, nb::arg("i")) \
            .def("simplex", [](const FrFiltration_##DIM##D& fil, size_t i) { \
                    return fatten_simplex_from_fr<oin_int, DIM, oin_real>(fil.get_cell(i), fil.geometry()); \
                }, nb::arg("i")) \
            .def_prop_ro("dim_first", &FrFiltration_##DIM##D::dims_first) \
            .def_prop_ro("dim_last", &FrFiltration_##DIM##D::dims_last) \
            .def("sorting_permutation", &FrFiltration_##DIM##D::get_sorting_permutation) \
            .def("inv_sorting_permutation", &FrFiltration_##DIM##D::get_inv_sorting_permutation) \
            /* uid accessors take the universal COMBINATORIAL uid that a (materialized fat) */ \
            /* cell carries; fr_slim_uid_from_comb_uid decodes it to vertices and re-keys it */ \
            /* into the slim (anchor,type) uid that this filtration is indexed by. */ \
            .def("value_by_uid", [](const FrFiltration_##DIM##D& fil, unsigned __int128 uid) { \
                    return fil.value_by_uid(fr_slim_uid_from_comb_uid<oin_int, DIM>(fil.geometry(), uid)); \
                }, nb::arg("uid")) \
            .def("sorted_id_by_uid", [](const FrFiltration_##DIM##D& fil, unsigned __int128 uid) { \
                    return fil.get_sorted_id_by_uid(fr_slim_uid_from_comb_uid<oin_int, DIM>(fil.geometry(), uid)); \
                }, nb::arg("uid")) \
            .def("cell_by_uid", [](const FrFiltration_##DIM##D& fil, unsigned __int128 uid) { \
                    oin_int slim_uid = fr_slim_uid_from_comb_uid<oin_int, DIM>(fil.geometry(), uid); \
                    return fatten_simplex_from_fr<oin_int, DIM, oin_real>(fil.get_cell_by_uid(slim_uid), fil.geometry()); \
                }, nb::arg("uid")) \
            .def("boundary_matrix", &FrFiltration_##DIM##D::boundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("boundary_matrix_in_dimension", &FrFiltration_##DIM##D::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("coboundary_matrix", &FrFiltration_##DIM##D::coboundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            /* boundary_matrix_rel is intentionally omitted: its relative path uses the */ \
            /* vector boundary(geom), which the slim Simplex<Int,Enc> wrapper does not */ \
            /* expose (only boundary_into). Relative homology with slim Freudenthal cells */ \
            /* awaits the buffer (_into) conversion of that path (deferred seam). */ \
            .def("star_closure", &FrFiltration_##DIM##D::star_closure, nb::arg("seed_sorted_ids"), nb::arg("n_threads")=1, \
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(), \
                    "Coface up-closure (union of stars) of the given cells (sorted_ids).") \
            .def("is_up_closed", &FrFiltration_##DIM##D::is_up_closed, nb::arg("cells"), nb::arg("n_threads")=1, \
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(), \
                    "True if the cells (sorted_ids) are closed under cofaces.") \
            .def("without_cells", &FrFiltration_##DIM##D::without_cells, nb::arg("cells_to_remove"), nb::rv_policy::move, \
                    "Subfiltration with the given cells (sorted_ids) removed; survivors keep order.") \
            .def("reset_ids_to_sorted_ids", &FrFiltration_##DIM##D::reset_ids_to_sorted_ids) \
            .def("set_values", &FrFiltration_##DIM##D::set_values, nb::arg("new_values"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def("__repr__", [](const FrFiltration_##DIM##D& fil) { \
                    std::stringstream ss; \
                    ss << fil; \
                    return ss.str(); \
                }) \
            .def_prop_rw("kind", &FrFiltration_##DIM##D::kind, &FrFiltration_##DIM##D::set_kind, \
                "FiltrationKind tag set by the constructor that built this filtration " \
                "(or User for hand-built ones).") \
            .def("__getstate__", [](const FrFiltration_##DIM##D& fil) -> FrFiltration_##DIM##DStateTuple { \
                      auto fat = fatten_all_fr<oin_int, DIM, oin_real>(fil); \
                      return std::make_tuple(fil.negate_, std::move(fat), fil.is_subfiltration_, \
                          fil.uid_to_sorted_id, fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_, \
                          fil.kind_, fil.geometry_.domain); \
                    }) \
            .def("__setstate__", [](FrFiltration_##DIM##D& fil, const FrFiltration_##DIM##DStateTuple& t) { \
                new (&fil) FrFiltration_##DIM##D(); \
                fil.negate_ = std::get<0>(t); \
                fil.is_subfiltration_ = std::get<2>(t); \
                fil.uid_to_sorted_id = std::get<3>(t); \
                fil.id_to_sorted_id_ = std::get<4>(t); \
                fil.sorted_id_to_id_ = std::get<5>(t); \
                fil.dim_first_ = std::get<6>(t); \
                fil.dim_last_ = std::get<7>(t); \
                fil.kind_ = std::get<8>(t); \
                /* FrGeometry is not picklable; rebuild it from the stored GridDomain, */ \
                /* then slim the fat cells back against it and rebuild the flat uid index */ \
                oin::FrGeometry<oin_int, DIM> frgeom(std::get<9>(t)); \
                fil.set_geometry(frgeom); \
                const auto& fat = std::get<1>(t); \
                typename FrFiltration_##DIM##D::CellVector slim; \
                slim.reserve(fat.size()); \
                for (const auto& fc : fat) slim.push_back(slim_simplex_from_fr<oin_int, DIM, oin_real>(fc, frgeom)); \
                fil.cells_ = std::move(slim); \
                fil.rebuild_uid_index_(); \
            }) \


    BIND_FR_FILTRATION(1);
    BIND_FR_FILTRATION(2);
    BIND_FR_FILTRATION(3);

    #undef BIND_FR_FILTRATION

    // ============ PackedSimplexFiltration (bit-packed VR/alpha) bindings ============
    // The bit-packed filtration stores Simplex<Int,BitPacked<Int,Word>> cells (sorted
    // vertex ids packed into one Word) + a shared PackedGeom{int bits}; every accessor
    // materializes the fat Python simplex (the "Simplex" class) via
    // fatten_simplex_from_packed. Factory-produced (vr_filtration(packed=True), and the
    // same macro is what a future alpha packed path would reuse); a fat simplex carries
    // no PackedGeom, so there is no public ctor-from-cells. One macro, two word tiers
    // (64-bit, 128-bit). Differences from the Freudenthal macro: the geometry is a
    // trivially-copyable int (pickle stores the bits directly, no geometry rebuild), and
    // the uid_to_sorted_id pickle field is omitted. The uid-keyed accessors
    // (value_by_uid/sorted_id_by_uid/cell_by_uid) take the universal COMBINATORIAL uid
    // (the 128-bit identity a materialized fat Simplex carries -- it crosses the Python
    // boundary via uid128_caster.h): packed_word_uid_from_comb_uid decodes it to vertices
    // and re-packs them into the internal Word uid for the hash lookup, so the caller uses
    // the same uid they read off a cell (matching the cube/Freudenthal/fat contract). The
    // uid_to_sorted_id map is dropped from the pickle because rebuild_uid_index_
    // regenerates the hash on unpickle (BitPacked is UsesDenseUidIndex=false), making the
    // stored map redundant.
    #define BIND_PACKED_FILTRATION(WORD, SUFFIX) \
        using PackedFiltration_##SUFFIX = oin::Filtration<oin::Simplex<oin_int, oin::BitPacked<oin_int, WORD>>, oin_real>; \
        using PackedFatSimplexValue_##SUFFIX = oin::CellWithValue<oin::Simplex<oin_int>, oin_real>; \
        using PackedFiltration_##SUFFIX##StateTuple = std::tuple<decltype(PackedFiltration_##SUFFIX::negate_), \
                                                std::vector<PackedFatSimplexValue_##SUFFIX>, \
                                                decltype(PackedFiltration_##SUFFIX::is_subfiltration_), \
                                                decltype(PackedFiltration_##SUFFIX::id_to_sorted_id_), \
                                                decltype(PackedFiltration_##SUFFIX::sorted_id_to_id_), \
                                                decltype(PackedFiltration_##SUFFIX::dim_first_), \
                                                decltype(PackedFiltration_##SUFFIX::dim_last_), \
                                                decltype(PackedFiltration_##SUFFIX::kind_), \
                                                int \
                                               >; \
        nb::class_<PackedFiltration_##SUFFIX>(m, "PackedSimplexFiltration_" #SUFFIX) \
            .def("__len__", &PackedFiltration_##SUFFIX::size) \
            .def("__iter__", [](const PackedFiltration_##SUFFIX& fil) -> nb::object { \
                    nb::object lst = nb::cast(fatten_all_packed<oin_int, WORD, oin_real>(fil)); \
                    return lst.attr("__iter__")(); \
                }) \
            .def("__getitem__", [](const PackedFiltration_##SUFFIX& fil, int i) { \
                    if (i < 0) i = fil.size() + i; \
                    return fatten_simplex_from_packed<oin_int, WORD, oin_real>(fil.get_cell(i), fil.geometry()); \
                }) \
            .def_prop_ro("negate", &PackedFiltration_##SUFFIX::negate) \
            .def("infinity", &PackedFiltration_##SUFFIX::infinity, "filtration-order +inf") \
            .def("neg_infinity", &PackedFiltration_##SUFFIX::neg_infinity, "filtration-order -inf") \
            .def("fil_min", &PackedFiltration_##SUFFIX::fil_min, nb::arg("a"), nb::arg("b"), "filtration-order min") \
            .def("fil_max", &PackedFiltration_##SUFFIX::fil_max, nb::arg("a"), nb::arg("b"), "filtration-order max") \
            .def_prop_ro("max_dim", &PackedFiltration_##SUFFIX::max_dim, "maximal dimension of a cell in filtration") \
            .def("cells", [](const PackedFiltration_##SUFFIX& fil) { return fatten_all_packed<oin_int, WORD, oin_real>(fil); }, \
                    "copy of all cells in filtration order") \
            .def("simplices", [](const PackedFiltration_##SUFFIX& fil) { return fatten_all_packed<oin_int, WORD, oin_real>(fil); }, \
                    "copy of all cells in filtration order") \
            .def("size", &PackedFiltration_##SUFFIX::size, "number of cells in filtration") \
            .def("size_in_dimension", &PackedFiltration_##SUFFIX::size_in_dimension, nb::arg("dim"), "number of cells of dimension dim") \
            .def("n_vertices", &PackedFiltration_##SUFFIX::n_vertices) \
            .def("cell_value_by_sorted_id", &PackedFiltration_##SUFFIX::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("simplex_value_by_sorted_id", &PackedFiltration_##SUFFIX::value_by_sorted_id, nb::arg("sorted_id")) \
            .def("id_by_sorted_id", &PackedFiltration_##SUFFIX::get_id_by_sorted_id, nb::arg("sorted_id")) \
            .def("sorted_id_by_id", &PackedFiltration_##SUFFIX::get_sorted_id, nb::arg("id")) \
            .def("cell", [](const PackedFiltration_##SUFFIX& fil, size_t i) { \
                    return fatten_simplex_from_packed<oin_int, WORD, oin_real>(fil.get_cell(i), fil.geometry()); \
                }, nb::arg("i")) \
            .def("simplex", [](const PackedFiltration_##SUFFIX& fil, size_t i) { \
                    return fatten_simplex_from_packed<oin_int, WORD, oin_real>(fil.get_cell(i), fil.geometry()); \
                }, nb::arg("i")) \
            .def_prop_ro("dim_first", &PackedFiltration_##SUFFIX::dims_first) \
            .def_prop_ro("dim_last", &PackedFiltration_##SUFFIX::dims_last) \
            .def("sorting_permutation", &PackedFiltration_##SUFFIX::get_sorting_permutation) \
            .def("inv_sorting_permutation", &PackedFiltration_##SUFFIX::get_inv_sorting_permutation) \
            .def("boundary_matrix", &PackedFiltration_##SUFFIX::boundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("boundary_matrix_in_dimension", &PackedFiltration_##SUFFIX::boundary_matrix_in_dimension, nb::arg("dim"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def("coboundary_matrix", &PackedFiltration_##SUFFIX::coboundary_matrix, nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            /* boundary_matrix_rel omitted (slim wrapper has no vector boundary(geom)); */ \
            /* uid accessors take the combinatorial uid and re-pack to the Word uid (see header) */ \
            .def("value_by_uid", [](const PackedFiltration_##SUFFIX& fil, unsigned __int128 uid) { \
                    return fil.value_by_uid(packed_word_uid_from_comb_uid<oin_int, WORD>(fil.geometry(), uid)); \
                }, nb::arg("uid")) \
            .def("sorted_id_by_uid", [](const PackedFiltration_##SUFFIX& fil, unsigned __int128 uid) { \
                    return fil.get_sorted_id_by_uid(packed_word_uid_from_comb_uid<oin_int, WORD>(fil.geometry(), uid)); \
                }, nb::arg("uid")) \
            .def("cell_by_uid", [](const PackedFiltration_##SUFFIX& fil, unsigned __int128 uid) { \
                    WORD w = packed_word_uid_from_comb_uid<oin_int, WORD>(fil.geometry(), uid); \
                    return fatten_simplex_from_packed<oin_int, WORD, oin_real>(fil.get_cell_by_uid(w), fil.geometry()); \
                }, nb::arg("uid")) \
            .def("star_closure", &PackedFiltration_##SUFFIX::star_closure, nb::arg("seed_sorted_ids"), nb::arg("n_threads")=1, \
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(), \
                    "Coface up-closure (union of stars) of the given cells (sorted_ids).") \
            .def("is_up_closed", &PackedFiltration_##SUFFIX::is_up_closed, nb::arg("cells"), nb::arg("n_threads")=1, \
                    nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>(), \
                    "True if the cells (sorted_ids) are closed under cofaces.") \
            .def("without_cells", &PackedFiltration_##SUFFIX::without_cells, nb::arg("cells_to_remove"), nb::rv_policy::move, \
                    "Subfiltration with the given cells (sorted_ids) removed; survivors keep order.") \
            .def("reset_ids_to_sorted_ids", &PackedFiltration_##SUFFIX::reset_ids_to_sorted_ids) \
            .def("set_values", &PackedFiltration_##SUFFIX::set_values, nb::arg("new_values"), nb::arg("n_threads")=1, nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()) \
            .def(nb::self == nb::self) \
            .def(nb::self != nb::self) \
            .def("__repr__", [](const PackedFiltration_##SUFFIX& fil) { \
                    std::stringstream ss; \
                    ss << fil; \
                    return ss.str(); \
                }) \
            .def_prop_rw("kind", &PackedFiltration_##SUFFIX::kind, &PackedFiltration_##SUFFIX::set_kind, \
                "FiltrationKind tag set by the constructor that built this filtration " \
                "(or User for hand-built ones).") \
            .def("__getstate__", [](const PackedFiltration_##SUFFIX& fil) -> PackedFiltration_##SUFFIX##StateTuple { \
                      auto fat = fatten_all_packed<oin_int, WORD, oin_real>(fil); \
                      return std::make_tuple(fil.negate_, std::move(fat), fil.is_subfiltration_, \
                          fil.id_to_sorted_id_, fil.sorted_id_to_id_, fil.dim_first_, fil.dim_last_, \
                          fil.kind_, fil.geometry_.bits); \
                    }) \
            .def("__setstate__", [](PackedFiltration_##SUFFIX& fil, const PackedFiltration_##SUFFIX##StateTuple& t) { \
                new (&fil) PackedFiltration_##SUFFIX(); \
                fil.negate_ = std::get<0>(t); \
                fil.is_subfiltration_ = std::get<2>(t); \
                fil.id_to_sorted_id_ = std::get<3>(t); \
                fil.sorted_id_to_id_ = std::get<4>(t); \
                fil.dim_first_ = std::get<5>(t); \
                fil.dim_last_ = std::get<6>(t); \
                fil.kind_ = std::get<7>(t); \
                /* PackedGeom is a trivial int width; rebuild it, slim the fat cells, and */ \
                /* regenerate the hash uid index (the packed cell uses the hash, not flat) */ \
                oin::PackedGeom geom{std::get<8>(t)}; \
                fil.set_geometry(geom); \
                const auto& fat = std::get<1>(t); \
                typename PackedFiltration_##SUFFIX::CellVector slim; \
                slim.reserve(fat.size()); \
                for (const auto& fc : fat) slim.push_back(slim_simplex_from_packed<oin_int, WORD, oin_real>(fc, geom)); \
                fil.cells_ = std::move(slim); \
                fil.rebuild_uid_index_(); \
            }) \


    BIND_PACKED_FILTRATION(std::uint64_t, 64);
    BIND_PACKED_FILTRATION(unsigned __int128, 128);

    #undef BIND_PACKED_FILTRATION

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
    // No slim FreudenthalFiltration overload: oineus.diff.min_filtration keys the
    // result back into the source filtrations by the materialized fat cell's
    // combinatorial uid, which a slim filtration (keyed by the (anchor,type) uid)
    // cannot resolve. Wiring it awaits the unified uid contract; until then
    // min_filtration over slim filtrations fails cleanly at overload resolution.

}
