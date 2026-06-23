#pragma once

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>

#define OINEUS_MAX_CUBE_DIM 3

#include "common_defs.h"
#include "log_wrapper.h"
#include "grid_domain.h"

#ifdef OINEUS_PYTHON_FRIENDS
namespace nanobind { class module_; }
void init_oineus_cells(nanobind::module_&);
#endif


namespace oineus {
    namespace cube_private {
        template<typename Int>
        Int get_vertex_part(Int cube_id) {
            return cube_id >> OINEUS_MAX_CUBE_DIM;
        }

        template<typename Int>
        dim_type get_dim(Int cube_id) {
            dim_type res = 0;
            for (int d = 0; d < OINEUS_MAX_CUBE_DIM; ++d) {
                if (cube_id & (1 << d))
                    res++;
            }
            return res;
        }

        template<typename Int, unsigned D>
        std::vector<std::array<Int, D>> get_cube_vertices(Int cube_id, const GridDomain<Int, D>& global_domain) {
            using Point = typename GridDomain<Int, D>::GridPoint;
            std::vector<Point> vertices;

            Int vertex_id = get_vertex_part(cube_id);

            Point vertex = global_domain.id_to_point(vertex_id);

            // displacements = standard basis vectors whose bits are set in cube_id
            std::vector<Point> disps;

            for (dim_type d = 0; d < OINEUS_MAX_CUBE_DIM; ++d) {
                if (cube_id & (1 << d)) {
                    Point pt = GridDomain<Int, D>::origin();
                    pt[d] = 1;
                    disps.push_back(pt);
                }
            }

            int n_vertices = (1 << disps.size());

            // vertex is sum of basis vectors belonging to an arbitrary subset of disps
            // we need to iterate over all subsets, vertex_idx = subset index
            for (int vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
                Point v = vertex;
                for (size_t disp_idx = 0; disp_idx < disps.size(); ++disp_idx) {
                    if (vertex_idx & (1 << disp_idx)) {
                        v += disps[disp_idx];
                    }
                }
                vertices.push_back(v);
            }

            return vertices;
        }

        // ---- packed-uid (co)boundary, the single combinatorial source ----
        // These operate on the uid alone plus the SHARED domain; they never touch
        // a per-cell domain copy (the slim cube does not carry one). Both the slim
        // Cube methods and the fat FatCube delegate here, so the boundary logic
        // lives in exactly one place. The face uids are returned in the order
        // produced by the construction below; callers that need them sorted (the
        // filtration boundary matrix) sort afterwards.

        template<typename Int, unsigned D>
        std::vector<Int> cube_boundary(Int uid, const GridDomain<Int, D>& dom)
        {
            using Point = typename GridDomain<Int, D>::GridPoint;
            std::vector<Int> result;
            std::vector<int> disps;

            for (int d = 0; d < OINEUS_MAX_CUBE_DIM; ++d)
                if (uid & (1 << d))
                    disps.push_back(d);

            // faces obtained by dropping one displacement from the current vertex
            for (size_t k = 0; k < disps.size(); ++k) {
                Int face_id = uid & ~(Int(1) << disps[k]);
                result.push_back(face_id);
            }

            // faces obtained by moving the vertex along one displacement and
            // dropping that displacement
            for (size_t k = 0; k < disps.size(); ++k) {
                Point p = dom.id_to_point(get_vertex_part(uid));
                p[disps[k]] += 1;
                Int face_id = dom.point_to_id(p) << OINEUS_MAX_CUBE_DIM;
                for (size_t j = 0; j < disps.size(); ++j)
                    if (j != k)
                        face_id |= (Int(1) << disps[j]);
                result.push_back(face_id);
            }

            return result;
        }

        template<typename Int, unsigned D>
        std::vector<Int> cube_coboundary(Int uid, const GridDomain<Int, D>& dom)
        {
            using Point = typename GridDomain<Int, D>::GridPoint;
            std::vector<Int> result;
            Int vertex_id = get_vertex_part(uid);
            Point vertex = dom.id_to_point(vertex_id);
            Int cube_bits = uid & ((1 << OINEUS_MAX_CUBE_DIM) - 1);

            // Case 1: add a basis vector without moving the vertex
            for (unsigned d = 0; d < D; ++d) {
                if (!(cube_bits & (1 << d))) {
                    Point opposite_corner = vertex;
                    for (unsigned dd = 0; dd < D; ++dd)
                        if ((cube_bits & (1 << dd)) || dd == d)
                            opposite_corner[dd] += 1;
                    if (dom.contains(opposite_corner))
                        result.push_back((vertex_id << OINEUS_MAX_CUBE_DIM) | cube_bits | (1 << d));
                }
            }

            // Case 2: move the vertex backward along the missing dimension, then
            // add that basis vector
            for (unsigned d = 0; d < D; ++d) {
                if (!(cube_bits & (1 << d))) {
                    Point shifted_vertex = vertex;
                    shifted_vertex[d] -= 1;
                    if (dom.contains(shifted_vertex)) {
                        Point opposite_corner = shifted_vertex;
                        for (unsigned dd = 0; dd < D; ++dd)
                            if ((cube_bits & (1 << dd)) || dd == d)
                                opposite_corner[dd] += 1;
                        if (dom.contains(opposite_corner)) {
                            Int shifted_vertex_id = dom.point_to_id(shifted_vertex);
                            result.push_back((shifted_vertex_id << OINEUS_MAX_CUBE_DIM) | cube_bits | (1 << d));
                        }
                    }
                }
            }

            return result;
        }

        template<typename Int, unsigned D>
        std::vector<Int> cube_top_cofaces(Int uid, const GridDomain<Int, D>& dom)
        {
            using Point = typename GridDomain<Int, D>::GridPoint;
            std::vector<Int> result;

            if (get_dim(uid) == D) {
                result.push_back(uid);
                return result;
            }

            Point vertex = dom.id_to_point(get_vertex_part(uid));
            Int cube_bits = uid & ((1 << OINEUS_MAX_CUBE_DIM) - 1);

            std::vector<int> missing_dims;
            for (unsigned d = 0; d < D; ++d)
                if (!(cube_bits & (1 << d)))
                    missing_dims.push_back(d);

            int dims_to_add = missing_dims.size();

            for (int shift_mask = 0; shift_mask < (1 << dims_to_add); ++shift_mask) {
                Point shifted_vertex = vertex;
                for (int i = 0; i < dims_to_add; ++i)
                    if (shift_mask & (1 << i))
                        shifted_vertex[missing_dims[i]] -= 1;

                if (dom.contains(shifted_vertex)) {
                    Point opposite_corner = shifted_vertex;
                    for (unsigned d = 0; d < D; ++d)
                        opposite_corner[d] += 1;
                    if (dom.contains(opposite_corner)) {
                        Int shifted_vertex_id = dom.point_to_id(shifted_vertex);
                        Int all_dim_bits = (1 << D) - 1;
                        result.push_back((shifted_vertex_id << OINEUS_MAX_CUBE_DIM) | all_dim_bits);
                    }
                }
            }
            return result;
        }
    }

    template<typename Int_, unsigned D>
    class FatCube;

    // Slim cubical cell: a single packed uid (anchor vertex id << 3 | face-bits)
    // plus a user id. It does NOT store the GridDomain; the geometry is owned once
    // by the Filtration (or supplied by the caller) and passed into the
    // (co)boundary methods. This keeps the per-cell footprint tiny (uid + user_id,
    // no per-cell domain copy), which is the cache win that the whole filtration is
    // built on. Equality/hash are uid-only: all cells in one filtration share the
    // same geometry, so the uid alone identifies a cube. For a self-contained cube
    // that carries its own domain (Python boundary, standalone use), see FatCube.
    template<typename Int_, unsigned D>
    class Cube {
    public:
        static_assert(D <= OINEUS_MAX_CUBE_DIM, "Cube ambient dimension D exceeds maximum supported dimension");

        using Int = Int_;
        using Uid = Int_;
        using Domain = GridDomain<Int, D>;
        using Geometry = Domain;
        using Point = typename Domain::GridPoint;
        using UidHasher = std::hash<Int>;
        using UidSet = std::unordered_set<Uid, UidHasher>;
        using Boundary = std::vector<Uid>;

        Cube() = default;
        Cube(const Cube&) = default;
        Cube(Cube&&) noexcept = default;
        Cube& operator=(const Cube&) = default;
        Cube& operator=(Cube&&) noexcept = default;

        explicit Cube(Int _id) : id_(_id) {}

        Cube(const Point& vertex, const std::vector<Int>& spanning_dims, const Domain& domain)
        {
            Uid vertex_part = domain.point_to_id(vertex) << OINEUS_MAX_CUBE_DIM;
            Uid face_part = 0;
            for(auto dim : spanning_dims) {
                face_part |= (1 << dim);
            }
            id_ = face_part | vertex_part;
        }

        dim_type dim() const { return cube_private::get_dim(id_); }

        Int get_uid() const { return id_; }
        // uid is set during construction, nothing to do here
        // still has to be provided
        void set_uid() { throw std::runtime_error("Changing UID of a cube is prohibited."); }
        Int get_id() const { return user_id_; }
        void set_id(Int user_id) { user_id_ = user_id; }

        // (co)boundary in uid space, computed against the shared domain. The
        // filtration owns the domain and passes it in; the cube itself stays slim.
        std::vector<Int> boundary(const Domain& dom) const { return cube_private::cube_boundary<Int, D>(id_, dom); }
        std::vector<Int> coboundary(const Domain& dom) const { return cube_private::cube_coboundary<Int, D>(id_, dom); }
        std::vector<Int> top_cofaces(const Domain& dom) const { return cube_private::cube_top_cofaces<Int, D>(id_, dom); }

        Point anchor_vertex(const Domain& dom) const
        {
            return dom.id_to_point(cube_private::get_vertex_part(id_));
        }

        std::vector<std::array<Int, D>> get_vertices(const Domain& dom) const
        {
            return cube_private::get_cube_vertices<Int, D>(id_, dom);
        }

        bool operator==(const Cube& other) const { return id_ == other.id_; }
        bool operator!=(const Cube& other) const { return !(*this == other); }

        // Human-readable form (operator<< below). A slim cube has no domain, so it
        // cannot show its vertices -- the uid and dimension are all it can print.
        std::string pretty_print() const
        {
            std::stringstream ss;
            ss << "Cube(uid=" << id_ << ", user_id=" << user_id_ << ", dim=" << dim() << ")";
            return ss.str();
        }

#ifdef OINEUS_PYTHON_FRIENDS
         friend void ::init_oineus_cells(nanobind::module_&);
#endif
         friend class FatCube<Int_, D>;

    private:
        static constexpr Int k_invalid_id = Int(-1);

        Int id_{k_invalid_id};
        Int user_id_{k_invalid_id};
    };

    template<typename Int, unsigned D>
    std::ostream &operator<<(std::ostream& out, const Cube<Int, D> &cube)
    {
        out << cube.pretty_print();
        return out;
    }

    // Fat, self-contained cube: a slim Cube bundled with a copy of its GridDomain.
    // This is the "materialized" form handed to Python (and usable standalone),
    // reproducing the historical Cube surface: no-argument (co)boundary, vertices,
    // anchor_vertex, domain, equality/hash that include the domain, and pickle. The
    // Filtration never stores these -- it stores slim Cubes and materializes a
    // FatCube on demand (Filtration::geometry() supplies the domain).
    template<typename Int_, unsigned D>
    class FatCube {
    public:
        static_assert(D <= OINEUS_MAX_CUBE_DIM, "Cube ambient dimension D exceeds maximum supported dimension");

        using Int = Int_;
        using Uid = Int_;
        using Domain = GridDomain<Int, D>;
        // Self-contained: carries its own domain, so as a cell it needs no external
        // geometry. The no-argument (co)boundary methods below use the stored domain.
        using Geometry = NoGeometry;
        using Point = typename Domain::GridPoint;
        using UidHasher = std::hash<Int>;
        using UidSet = std::unordered_set<Uid, UidHasher>;
        using Boundary = std::vector<Uid>;

        FatCube() = default;
        FatCube(const FatCube&) = default;
        FatCube(FatCube&&) noexcept = default;
        FatCube& operator=(const FatCube&) = default;
        FatCube& operator=(FatCube&&) noexcept = default;

        FatCube(Int _id, const Domain& domain) : cube_(_id), global_domain_(domain) {}

        FatCube(const Point& vertex, const std::vector<Int>& spanning_dims, const Domain& domain)
                : cube_(vertex, spanning_dims, domain), global_domain_(domain) {}

        FatCube(const Cube<Int, D>& cube, const Domain& domain) : cube_(cube), global_domain_(domain) {}

        Domain global_domain() const { return global_domain_; }

        dim_type dim() const { return cube_.dim(); }

        Int get_uid() const { return cube_.get_uid(); }
        void set_uid() { cube_.set_uid(); }
        Int get_id() const { return cube_.get_id(); }
        void set_id(Int user_id) { cube_.set_id(user_id); }

        // self-contained boundary as uids (the cell concept's boundary()): uses the
        // stored domain. CellWithValue<FatCube> forwards here for the NoGeometry case.
        Boundary boundary() const { return cube_.boundary(global_domain_); }
        Boundary coboundary() const { return cube_.coboundary(global_domain_); }
        Boundary top_cofaces() const { return cube_.top_cofaces(global_domain_); }

        std::vector<FatCube> boundary_cubes() const { return materialize(cube_.boundary(global_domain_)); }
        std::vector<FatCube> coboundary_cubes() const { return materialize(cube_.coboundary(global_domain_)); }
        std::vector<FatCube> top_cofaces_cubes() const { return materialize(cube_.top_cofaces(global_domain_)); }

        bool operator==(const FatCube &other) const { return get_uid() == other.get_uid() && global_domain_ == other.global_domain_; }
        bool operator!=(const FatCube &other) const { return !(*this == other); }

        Point anchor_vertex() const { return cube_.anchor_vertex(global_domain_); }

        std::vector<std::array<Int, D>> get_vertices() const { return cube_.get_vertices(global_domain_); }

        std::string pretty_print() const
        {
            std::stringstream ss;
            ss << "Cube([";
            auto vs = get_vertices();
            for(size_t i = 0; i < vs.size() - 1; ++i)
                ss << vs[i] << ", ";
            ss << vs[vs.size() - 1] << "])";
            return ss.str();
        }

        std::string repr_print() const
        {
            std::stringstream ss;
            ss << "Cube([";
            ss << "uid=" << get_uid() << ", user_id = " << get_id() << ", domain = " << global_domain_ << ",";
            ss << "anchor = " << anchor_vertex() << ", vertices=[";
            auto vs = get_vertices();
            for(size_t i = 0; i < vs.size() - 1; ++i)
                ss << vs[i] << ", ";
            ss << vs[vs.size() - 1] << "])";
            return ss.str();
        }

#ifdef OINEUS_PYTHON_FRIENDS
         friend void ::init_oineus_cells(nanobind::module_&);
#endif

    private:
        std::vector<FatCube> materialize(const std::vector<Int>& uids) const
        {
            std::vector<FatCube> result;
            result.reserve(uids.size());
            for(auto uid : uids)
                result.emplace_back(uid, global_domain_);
            return result;
        }

        Cube<Int, D> cube_;
        Domain global_domain_;
    };

    template<typename Int, unsigned D>
    std::ostream &operator<<(std::ostream& out, const FatCube<Int, D> &cube)
    {
        out << cube.pretty_print();
        return out;
    }
} // namespace oineus

namespace std {
    template<typename Int, unsigned D>
    struct hash<oineus::Cube<Int, D>> {
        size_t operator()(const oineus::Cube<Int, D>& cube) const {
            return std::hash<Int>{}(cube.get_uid());
        }
    };

    template<typename Int, unsigned D>
    struct hash<oineus::FatCube<Int, D>> {
        size_t operator()(const oineus::FatCube<Int, D>& cube) const {
            size_t seed = std::hash<Int>{}(cube.get_uid());

            // Combine with domain hash
            size_t domain_hash = std::hash<typename oineus::FatCube<Int, D>::Domain>{}(cube.global_domain());
            seed ^= domain_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}
