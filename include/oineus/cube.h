#pragma once

#include <array>
#include <ostream>
#include <functional>

#define OINEUS_MAX_CUBE_DIM 3

#include "common_defs.h"
#include "log_wrapper.h"
#include "grid_domain.h"

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
    }

    template<typename Int_, unsigned D>
    class Cube {
    public:
        static_assert(D <= OINEUS_MAX_CUBE_DIM, "Cube ambient dimension D exceeds maximum supported dimension");

        using Int = Int_;
        using Uid = Int_;
        using Domain = GridDomain<Int, D>;
        using Point = typename Domain::GridPoint;
        using UidHasher = std::hash<Int>;
        using UidSet = std::unordered_set<Uid, UidHasher>;
        using Boundary = std::vector<Uid>;

        Cube() = default;
        Cube(const Cube&) = default;
        Cube(Cube&&) noexcept = default;
        Cube& operator=(const Cube&) = default;
        Cube& operator=(Cube&&) noexcept = default;

        Cube(Int _id, const Domain& domain) : id_(_id), global_domain_(domain) {};

        Cube(const Point& vertex, const std::vector<Int>& spanning_dims, const Domain& domain) : global_domain_(domain)
        {
            Uid vertex_part = domain.point_to_id(vertex) << OINEUS_MAX_CUBE_DIM;
            Uid face_part = 0;
            for(auto dim : spanning_dims) {
                face_part |= (1 << dim);
            }
            id_ = face_part | vertex_part;
        }

        Domain global_domain() const { return global_domain_; }

        dim_type dim() const { return cube_private::get_dim(id_); }

        Int get_uid() const { return id_; }
        // uid is set during construction, nothing to do here
        // still has to be provided
        void set_uid() { throw std::runtime_error("Changing UID of a cube is prohibited."); }
        Int get_id() const { return user_id_; }
        void set_id(Int user_id) { user_id_ = user_id; }

        std::vector<Int> boundary() const
        {
            auto logger = spd::get("console");

            std::vector<Int> result;
            std::vector<int> disps;

            for (int d = 0; d < OINEUS_MAX_CUBE_DIM; ++d) {
                if (get_uid() & (1 << d)) {
                    disps.push_back(d);
                }
            }

            // add faces from current vertex

            for(size_t k = 0; k < disps.size(); ++k) {
                assert(get_uid() & (1 << disps[k]));
                // unset the disps[k] bit
                Int face_id = get_uid() & ~(1 << disps[k]);
                assert(::oineus::cube_private::get_vertex_part(get_uid()) == ::oineus::cube_private::get_vertex_part(face_id));
                assert(::oineus::cube_private::get_dim(get_uid()) == ::oineus::cube_private::get_dim(face_id) + 1);
                result.push_back(face_id);
            }

            // pick one displacement vector, move vertex along it
            // add face defined by the remaining vectors
            for (size_t k = 0; k < disps.size(); ++k) {
                Point p = anchor_vertex();
                p[disps[k]] += 1;
                Int face_id = global_domain_.point_to_id(p) << OINEUS_MAX_CUBE_DIM;
                // set the bits corresponding to other displacements
                for (size_t j = 0; j < disps.size(); ++j) {
                    if (j == k)
                        continue;
                    face_id |= (1 << disps[j]);
                }
                assert(cube_private::get_dim(get_uid()) == cube_private::get_dim(face_id) + 1);
                result.push_back(face_id);
            }

            return result;
        } // boundary

        std::vector<Int> coboundary() const
        {
            std::vector<Int> result;
            Int vertex_id = cube_private::get_vertex_part(get_uid());
            Point vertex = anchor_vertex();
            Int cube_bits = get_uid() & ((1 << OINEUS_MAX_CUBE_DIM) - 1);

            // Case 1: Add a basis vector without moving vertex
            for (unsigned d = 0; d < D; ++d) {
                if (!(cube_bits & (1 << d))) {  // if bit d is not set
                    // Check if the entire new cube fits in domain
                    Point opposite_corner = vertex;
                    for (unsigned dd = 0; dd < D; ++dd) {
                        if ((cube_bits & (1 << dd)) || dd == d) {  // existing dims + new dim
                            opposite_corner[dd] += 1;
                        }
                    }
                    if (global_domain_.contains(opposite_corner)) {
                        Int coface_id = (vertex_id << OINEUS_MAX_CUBE_DIM) | cube_bits | (1 << d);
                        result.push_back(coface_id);
                    }
                }
            }

            // Case 2: Move vertex backward along missing dimension and add that basis vector
            for (unsigned d = 0; d < D; ++d) {
                if (!(cube_bits & (1 << d))) {  // if bit d is not set
                    Point shifted_vertex = vertex;
                    shifted_vertex[d] -= 1;

                    if (global_domain_.contains(shifted_vertex)) {
                        // Check if the entire new cube fits in domain
                        Point opposite_corner = shifted_vertex;
                        for (unsigned dd = 0; dd < D; ++dd) {
                            if ((cube_bits & (1 << dd)) || dd == d) {  // existing dims + new dim
                                opposite_corner[dd] += 1;
                            }
                        }
                        if (global_domain_.contains(opposite_corner)) {
                            Int shifted_vertex_id = global_domain_.point_to_id(shifted_vertex);
                            Int coface_id = (shifted_vertex_id << OINEUS_MAX_CUBE_DIM) | cube_bits | (1 << d);
                            result.push_back(coface_id);
                        }
                    }
                }
            }

            return result;
        }

        std::vector<Int> top_cofaces() const
        {
            std::vector<Int> result;
            dim_type cube_dim = dim();

            if (cube_dim == D) {
                // Already top-dimensional
                result.push_back(get_uid());
                return result;
            }

            Point vertex = anchor_vertex();
            Int cube_bits = get_uid() & ((1 << OINEUS_MAX_CUBE_DIM) - 1);

            // Find which dimensions are missing
            std::vector<int> missing_dims;
            for (unsigned d = 0; d < D; ++d) {
                if (!(cube_bits & (1 << d))) {
                    missing_dims.push_back(d);
                }
            }

            int dims_to_add = missing_dims.size();

            // For each way to shift the vertex along the missing dimensions
            for (int shift_mask = 0; shift_mask < (1 << dims_to_add); ++shift_mask) {
                Point shifted_vertex = vertex;

                for (int i = 0; i < dims_to_add; ++i) {
                    if (shift_mask & (1 << i)) {
                        shifted_vertex[missing_dims[i]] -= 1;
                    }
                }

                // Check if the shifted vertex is in domain
                // Also need to check if the entire D-cube fits in the domain
                if (global_domain_.contains(shifted_vertex)) {
                    // Check if the opposite corner of the D-cube is also in domain
                    Point opposite_corner = shifted_vertex;
                    for (unsigned d = 0; d < D; ++d) {
                        opposite_corner[d] += 1;
                    }
                    if (global_domain_.contains(opposite_corner)) {
                        Int shifted_vertex_id = global_domain_.point_to_id(shifted_vertex);
                        Int all_dim_bits = (1 << D) - 1;
                        Int coface_id = (shifted_vertex_id << OINEUS_MAX_CUBE_DIM) | all_dim_bits;
                        result.push_back(coface_id);
                    }
                }
            }
            return result;
        }

        std::vector<Cube> coboundary_cubes() const
        {
            std::vector<Cube> result;
            for(auto cob_uid : coboundary()) {
                result.emplace_back(cob_uid, global_domain());
            }
            return result;
        }

        std::vector<Cube> boundary_cubes() const
        {
            std::vector<Cube> result;
            for(auto cob_uid : boundary()) {
                result.emplace_back(cob_uid, global_domain());
            }
            return result;
        }

        std::vector<Cube> top_cofaces_cubes() const
        {
            std::vector<Cube> result;
            for(auto cob_uid : top_cofaces()) {
                result.emplace_back(cob_uid, global_domain());
            }
            return result;
        }

        bool operator==(const Cube &other) const { return get_uid() == other.get_uid() && global_domain_ == other.global_domain_ ;}
        bool operator!=(const Cube &other) const { return !(*this == other);}

        Point anchor_vertex() const
        {
            // extract last OINEUS_MAX_CUBE_DIM bits of id
            Int v_idx = cube_private::get_vertex_part(id_);
            return global_domain_.id_to_point(v_idx);
        }

        std::vector<std::array<Int, D>> get_vertices() const { return cube_private::get_cube_vertices<Int, D>(get_uid(), global_domain()); }

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
        friend void init_oineus_cells(nb::module_&);
#endif

    // private:
        static constexpr Int k_invalid_id = Int(-1);

        Int id_{k_invalid_id};
        Int user_id_{k_invalid_id};
        Domain global_domain_;
    };

    template<typename Int, unsigned D>
    std::ostream &operator<<(std::ostream& out, const Cube<Int, D> &cube)
    {
        out << cube.pretty_print();
        return out;
    }
} // namespace oineus

namespace std {
    template<typename Int, unsigned D>
    struct hash<oineus::Cube<Int, D>> {
        size_t operator()(const oineus::Cube<Int, D>& cube) const {
            size_t seed = std::hash<Int>{}(cube.get_uid());

            // Combine with domain hash
            size_t domain_hash = std::hash<typename oineus::Cube<Int, D>::Domain>{}(cube.global_domain());
            seed ^= domain_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}
