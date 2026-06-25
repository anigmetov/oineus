#pragma once

#include <array>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include <utility>

#include "common_defs.h"
#include "grid_domain.h"

namespace oineus {

    namespace fr_private {
        template<typename Int, unsigned D>
        using Pt = std::array<Int, D>;

        // min corner (per-coordinate minimum) of a displacement set
        template<typename Int, unsigned D>
        Pt<Int, D> min_corner(const std::vector<Pt<Int, D>>& s)
        {
            Pt<Int, D> m = s[0];
            for (const auto& p : s)
                for (unsigned d = 0; d < D; ++d)
                    m[d] = std::min(m[d], p[d]);
            return m;
        }

        // translate a displacement set so its min corner is the origin, then sort:
        // a translation-invariant, order-invariant canonical key for a simplex
        // "type" (the shape of the Freudenthal simplex independent of where it sits).
        template<typename Int, unsigned D>
        std::vector<Pt<Int, D>> normalize(std::vector<Pt<Int, D>> s)
        {
            Pt<Int, D> m = min_corner<Int, D>(s);
            for (auto& p : s)
                for (unsigned d = 0; d < D; ++d)
                    p[d] -= m[d];
            std::sort(s.begin(), s.end());
            return s;
        }
    } // namespace fr_private

    // Shared geometry for a Freudenthal (Kuhn-triangulation) filtration on a D-grid:
    // the GridDomain plus the (anchor, type) (co)boundary tables, built once from the
    // displacement set. Unlike the cube's compile-time OINEUS_MAX_CUBE_DIM, the number
    // of simplex types is data-dependent, so type_bits (how many low bits of a uid
    // hold the type) is a runtime value living here, not on the cell. This struct is
    // the FreudenthalCell's Geometry: the Filtration owns one and threads it into the
    // cell's (co)boundary. Restricted to non-wrap grids -- the id-offset arithmetic
    // (facet_anchor_id = anchor_id + id_off) relies on point_to_id being linear.
    template<typename Int_, unsigned D>
    struct FrGeometry {
        using Int = Int_;
        using Domain = GridDomain<Int, D>;
        using Point = typename Domain::GridPoint;

        // boundary: cell (anchor_id, type t) -> facets {(anchor_id + id_off, ft)}
        struct BdEntry { Int id_off; int ft; };
        // coboundary: cell (anchor_id, type t) -> cofacets at anchor_point - delta,
        // valid iff it and its far corner (delta + maxoff) stay in the grid
        struct CobEntry { Point delta; Point maxoff; int ct; };

        Domain domain;
        std::vector<std::vector<BdEntry>> bd_table;   // indexed by type
        std::vector<std::vector<CobEntry>> cob_table; // indexed by type
        std::vector<int> type_dim;                    // dimension of each type
        // normalized displacement pattern of each type, relative to the cell's anchor
        // (= min-corner = min-id vertex of a Kuhn simplex). The inverse of set2type;
        // kept so a (anchor,type) uid can be materialized back to its grid vertex ids.
        std::vector<std::vector<Point>> type_disps;
        std::map<std::vector<Point>, int> set2type;   // normalized pattern -> type
        int type_bits {1};
        Int type_mask {1};

        FrGeometry() = default;

        explicit FrGeometry(const Domain& dom) : domain(dom)
        {
            // facet_anchor_id = anchor_id + id_off relies on point_to_id being linear
            // (no modular wrap-around), so the packed (anchor,type) form is only valid
            // on non-wrap grids. Wrap grids must use the Simplex filtration.
            if (dom.wrap())
                throw std::runtime_error("FreudenthalCell does not support wrap grids; use the Simplex filtration");
            build_tables();
        }

        bool operator==(const FrGeometry& o) const
        {
            return domain == o.domain and type_bits == o.type_bits and type_dim == o.type_dim;
        }
        bool operator!=(const FrGeometry& o) const { return not(*this == o); }

        // (anchor_id, type) <-> uid helpers
        Int make_uid(Int anchor_id, int type) const { return (anchor_id << type_bits) | static_cast<Int>(type); }
        Int anchor_of(Int uid) const { return uid >> type_bits; }
        int type_of(Int uid) const { return static_cast<int>(uid & type_mask); }
        dim_type dim_of_type(int type) const { return static_cast<dim_type>(type_dim[type]); }
        // dimension of the cell with this uid (its type's dimension). The cell stores
        // dim separately because dim() has no geometry; construct cells with this so
        // the stored dim can never desync from the uid.
        dim_type dim_of_uid(Int uid) const { return dim_of_type(type_of(uid)); }

        // (anchor, type) uid for the Freudenthal simplex on these vertex ids: the
        // anchor is the min-id (min-corner) vertex; the type is looked up from the
        // normalized displacement pattern. Used once per cell at construction (not on
        // the hot path) to encode a simplex into the compact form.
        Int uid_of_vertices(const std::vector<Int>& vertex_ids) const
        {
            Int anchor_id = *std::min_element(vertex_ids.begin(), vertex_ids.end());
            Point ap = domain.id_to_point(anchor_id);
            std::vector<Point> disps;
            disps.reserve(vertex_ids.size());
            for (Int v : vertex_ids) {
                Point p = domain.id_to_point(v);
                for (unsigned d = 0; d < D; ++d)
                    p[d] -= ap[d];
                disps.push_back(p);
            }
            auto it = set2type.find(fr_private::normalize<Int, D>(disps));
            if (it == set2type.end())
                throw std::runtime_error("uid_of_vertices: vertex set is not a Freudenthal simplex of this grid");
            return make_uid(anchor_id, it->second);
        }

        // Materialize the grid vertex ids of the cell with this uid: anchor point +
        // each of the type's displacements, mapped back to ids, sorted (the canonical
        // fat Simplex vertex order). Inverse of uid_of_vertices. This is the slim->fat
        // materialization the filtration's fat-cell accessors / the Fattener use; for
        // a Kuhn simplex the anchor is the min-corner, so the normalized type pattern
        // sits at the anchor and these additions land exactly on the vertices.
        std::vector<Int> vertices_of(Int uid) const
        {
            Point ap = domain.id_to_point(anchor_of(uid));
            const std::vector<Point>& disps = type_disps[type_of(uid)];
            std::vector<Int> vids;
            vids.reserve(disps.size());
            for (const Point& d : disps) {
                Point p;
                for (unsigned k = 0; k < D; ++k)
                    p[k] = ap[k] + d[k];
                vids.push_back(domain.point_to_id(p));
            }
            std::sort(vids.begin(), vids.end());
            return vids;
        }

    private:
        void build_tables()
        {
            using Point = typename Domain::GridPoint;
            std::vector<Point> type_maxoff;

            auto register_type = [&](const std::vector<Point>& norm, int d) {
                auto it = set2type.find(norm);
                if (it != set2type.end())
                    return it->second;
                int id = static_cast<int>(type_dim.size());
                set2type[norm] = id;
                type_dim.push_back(d);
                type_disps.push_back(norm);
                Point mo{};
                for (unsigned dd = 0; dd < D; ++dd) {
                    Int mx = 0;
                    for (const auto& p : norm)
                        mx = std::max(mx, p[dd]);
                    mo[dd] = mx;
                }
                type_maxoff.push_back(mo);
                return id;
            };

            // every face dimension contributes its displacement patterns as types
            for (int d = 0; d <= static_cast<int>(D); ++d)
                for (const auto& pattern : domain.get_fr_displacements(d)) {
                    std::vector<Point> s(pattern.begin(), pattern.end());
                    register_type(fr_private::normalize<Int, D>(s), d);
                }

            int n_types = static_cast<int>(type_dim.size());
            type_bits = 1;
            while ((1 << type_bits) < n_types)
                ++type_bits;
            type_mask = (Int(1) << type_bits) - 1;

            // boundary table: drop each vertex of a type, canonicalize the remaining
            // facet, record the new anchor's id offset + the facet type. The facet's
            // point offset m (its min corner relative to the parent anchor) is needed
            // to invert into the coboundary table, so collect (parent, ft, m) locally.
            struct CobPending { int parent; int ft; Point m; };
            std::vector<CobPending> cob_pending;

            bd_table.assign(n_types, {});
            for (int t = 0; t < n_types; ++t) {
                int d = type_dim[t];
                if (d == 0)
                    continue;
                for (int i = 0; i <= d; ++i) {
                    std::vector<Point> rem;
                    for (int j = 0; j <= d; ++j)
                        if (j != i)
                            rem.push_back(type_disps[t][j]);
                    Point m = fr_private::min_corner<Int, D>(rem);
                    int ft = set2type.at(fr_private::normalize<Int, D>(rem));
                    bd_table[t].push_back({domain.point_to_id(m), ft});
                    cob_pending.push_back({t, ft, m});
                }
            }

            // coboundary table: invert the boundary relation
            cob_table.assign(n_types, {});
            for (const auto& cp : cob_pending)
                cob_table[cp.ft].push_back({cp.m, type_maxoff[cp.parent], cp.parent});
        }
    };

    template<typename Int_, unsigned D>
    class FreudenthalCell {
    public:
        using Int = Int_;
        using Uid = Int_;
        using Geometry = FrGeometry<Int, D>;
        using Point = typename Geometry::Point;
        using UidHasher = std::hash<Int>;
        using UidSet = std::unordered_set<Uid, UidHasher>;
        using Boundary = std::vector<Uid>;

        FreudenthalCell() = default;
        FreudenthalCell(const FreudenthalCell&) = default;
        FreudenthalCell(FreudenthalCell&&) noexcept = default;
        FreudenthalCell& operator=(const FreudenthalCell&) = default;
        FreudenthalCell& operator=(FreudenthalCell&&) noexcept = default;

        // uid = anchor_id << type_bits | type; dim is stored because the cell concept
        // needs dim() without the geometry (sorting, dim ranges). The caller MUST pass
        // d == geometry.dim_of_uid(uid) -- prefer FrGeometry::dim_of_uid to derive it
        // so the stored dim cannot desync from the uid. The uid space is
        // (num_vertices << type_bits); Int must be wide enough to hold it (long int in
        // the Python build; int suffices only for small grids).
        FreudenthalCell(Int uid, dim_type d) : id_(uid), dim_(d) {}

        dim_type dim() const { return dim_; }

        Int get_uid() const { return id_; }
        void set_uid() { throw std::runtime_error("Changing UID of a FreudenthalCell is prohibited."); }
        Int get_id() const { return user_id_; }
        void set_id(Int user_id) { user_id_ = user_id; }

        // Buffer (co)boundary against the shared tables: invoke emit(face_uid) for
        // each (co)face, no intermediate vector. These are the alloc-elided bodies the
        // Filtration builders call; the vector-returning boundary() wraps the first.
        template<typename Visitor>
        void boundary_into(const Geometry& g, Visitor&& visit) const
        {
            assert(id_ != k_invalid_id);
            Int anchor_id = g.anchor_of(id_);
            int t = g.type_of(id_);
            for (const auto& e : g.bd_table[t])
                visit(((anchor_id + e.id_off) << g.type_bits) | static_cast<Int>(e.ft));
        }

        template<typename Visitor>
        void coboundary_into(const Geometry& g, Visitor&& visit) const
        {
            assert(id_ != k_invalid_id);
            Int anchor_id = g.anchor_of(id_);
            int t = g.type_of(id_);
            Point ap = g.domain.id_to_point(anchor_id);
            Point shape = g.domain.shape();
            for (const auto& e : g.cob_table[t]) {
                Point ca;
                bool ok = true;
                for (unsigned dd = 0; dd < D; ++dd) {
                    ca[dd] = ap[dd] - e.delta[dd];
                    if (ca[dd] < 0 or ca[dd] + e.maxoff[dd] > shape[dd] - 1) {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                    visit((g.domain.point_to_id(ca) << g.type_bits) | static_cast<Int>(e.ct));
            }
        }

        std::vector<Int> boundary(const Geometry& g) const
        {
            std::vector<Int> result;
            boundary_into(g, [&result](Int u) { result.push_back(u); });
            return result;
        }

        std::vector<Int> coboundary(const Geometry& g) const
        {
            std::vector<Int> result;
            coboundary_into(g, [&result](Int u) { result.push_back(u); });
            return result;
        }

        // The grid vertex ids of this cell (slim->fat materialization), delegating to
        // the geometry's per-type displacement patterns. The fat Simplex on these
        // vertices is the honest cell this slim (anchor,type) form stands in for.
        std::vector<Int> vertices(const Geometry& g) const { return g.vertices_of(id_); }

        bool operator==(const FreudenthalCell& other) const { return id_ == other.id_; }
        bool operator!=(const FreudenthalCell& other) const { return !(*this == other); }

        // A slim cell carries no geometry, so it can only print its uid and dim.
        std::string pretty_print() const
        {
            std::stringstream ss;
            ss << "FreudenthalCell(uid=" << id_ << ", user_id=" << user_id_ << ", dim=" << dim_ << ")";
            return ss.str();
        }

    private:
        static constexpr Int k_invalid_id = Int(-1);

        Int id_ {k_invalid_id};
        Int user_id_ {k_invalid_id};
        dim_type dim_ {0};
    };

    template<typename Int, unsigned D>
    std::ostream& operator<<(std::ostream& out, const FreudenthalCell<Int, D>& c)
    {
        out << c.pretty_print();
        return out;
    }

    // Dense integer uid (anchor << type_bits | type), with a table-driven buffer
    // (co)boundary: advertise the packed path so the Filtration uses the alloc-elided
    // builders and the flat uid->sorted_id index, exactly like Cube.
    template<typename Int, unsigned D>
    struct HasPackedBoundary<FreudenthalCell<Int, D>> : std::true_type {};
    template<typename Int, unsigned D>
    struct HasDirectCoboundary<FreudenthalCell<Int, D>> : std::true_type {};
    template<typename Int, unsigned D>
    struct UsesDenseUidIndex<FreudenthalCell<Int, D>> : std::true_type {};

} // namespace oineus

namespace std {
    template<typename Int, unsigned D>
    struct hash<oineus::FreudenthalCell<Int, D>> {
        size_t operator()(const oineus::FreudenthalCell<Int, D>& c) const
        {
            return std::hash<Int>{}(c.get_uid());
        }
    };
}
