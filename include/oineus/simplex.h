#pragma once

#include <cstdint>
#include <iomanip>
#include <limits>
#include <vector>
#include <ostream>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include <string>
#include <sstream>

#if !defined(__SIZEOF_INT128__)
#  error "oineus::Simplex::Uid uses unsigned __int128; this requires gcc/clang. " \
         "On MSVC or other compilers without __int128, supply a portable 128-bit " \
         "integer (e.g. boost::multiprecision::uint128_t) and adapt the hash / " \
         "operator<< definitions below."
#endif

//using namespace std::rel_ops;

namespace oineus {

// Neither libstdc++ nor libc++ ship operator<< for __int128, so provide one in
// our namespace. Format: 0x{hi:016x}{lo:016x} -- readable and unambiguous.
inline std::ostream& operator<<(std::ostream& os, unsigned __int128 v)
{
    auto saved_flags = os.flags();
    auto saved_fill  = os.fill();
    auto hi = static_cast<std::uint64_t>(v >> 64);
    auto lo = static_cast<std::uint64_t>(v);
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << hi
              << std::setw(16) << std::setfill('0') << lo;
    os.flags(saved_flags);
    os.fill(saved_fill);
    return os;
}

template<typename IntIn, typename IntOut>
IntOut comb(IntIn n, IntIn k)
{
    if (n < k || n == 0) {
        return static_cast<IntOut>(0);
    }

    if (k == 0 || k == n)
        return static_cast<IntOut>(1);

    IntOut result = 1;
    for(IntOut i = 1; i <= k; i++) {
        result *= (n - i + 1);
        assert(result % i == 0);
        result /= i;
    }
    return result;
}

// combinatorial simplex numbering, as in Ripser (see Bauer's paper).
// Encodes dimension info in the 4 most significant bits of the 128-bit result.
template<typename IntIn>
unsigned __int128 simplex_uid(const std::vector<IntIn>& vertices)
{
    using Uid = unsigned __int128;
    Uid dim_info = static_cast<Uid>(vertices.size() + 1) << (8 * sizeof(Uid) - 4);
    Uid uid = 0;
    for(IntIn i = 0; i < static_cast<IntIn>(vertices.size()); i++) {
        uid += comb<IntIn, Uid>(vertices[i], i + 1);
    }
    return uid | dim_info;
}

template<typename Int_>
struct VREdge {
    using Int = Int_;
    Int x;
    Int y;
    bool operator==(const VREdge& other) const { return x == other.x and y == other.y; }
    bool operator!=(const VREdge& other) const { return !(*this == other); }
    bool operator<(const VREdge& other) const { return x < other.x or (x == other.x and y < other.y); };
    bool operator>(const VREdge& other) const { return other < *this; }
    bool operator<=(const VREdge& other) const { return *this < other or *this == other; }
    bool operator>=(const VREdge& other) const { return *this > other or *this == other; }
};

template<class Int>
inline std::ostream& operator<<(std::ostream& out, const VREdge<Int>& e)
{
    out << "edge(x=" << e.x << ", y=" << e.y << ")";
    return out;
}

template<typename Int_>
struct Simplex {
    using Int = Int_;
    using IdxVector = std::vector<Int>;

    using Uid = unsigned __int128;
    // for Z2 only for now
    using Boundary = std::vector<Uid>;

    static constexpr Int k_invalid_id = Int(-1);

    Int id_ {k_invalid_id};
    Uid uid_ {};
    IdxVector vertices_;

    Simplex() = default;
    Simplex(const Simplex&) = default;
    Simplex(Simplex&&) noexcept = default;
    Simplex& operator=(const Simplex&) = default;
    Simplex& operator=(Simplex&&) noexcept = default;

    Simplex(const IdxVector& _vertices)
            :vertices_(_vertices)
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        if (vertices_.size() == 1)
            id_ = vertices_[0];
        else
            std::sort(vertices_.begin(), vertices_.end());

        set_uid();
    }

    // uids are set in parallel
    void set_uid() { uid_ = simplex_uid<Int>(vertices_); }

    dim_type dim() const { return static_cast<Int>(vertices_.size()) - 1; }

    Int get_id() const { return id_; }
    void set_id(Int new_id) { id_ = new_id; }

    const IdxVector& get_vertices() const { return vertices_; }

    Simplex(const Int _id, const IdxVector& _vertices)
            :id_(_id), vertices_(_vertices)
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        if (vertices_.size() > 1)
            std::sort(vertices_.begin(), vertices_.end());

        set_uid();
    }

    Boundary boundary() const
    {
        Boundary boundary;

        if (dim() == 0)
            return boundary;

        boundary.reserve(vertices_.size());
        // TODO: do not materialize tau and just skip in when computing uid?
        for(size_t i = 0 ; i < vertices_.size() ; ++i) {
            IdxVector tau;
            tau.reserve(vertices_.size() - 1);

            for(size_t j = 0 ; j < vertices_.size() ; ++j)
                if (j != i)
                    tau.push_back(vertices_[j]);

            // vertices_ is sorted -> tau is sorted automatically

            boundary.push_back(simplex_uid<Int>(tau));
        }

        return boundary;
    }

    // create a new simplex by joining with vertex and assign value to it
    Simplex join(Int new_id, Int vertex) const
    {
        // new vertex must not be present in this->vertices
        assert(std::find(vertices_.begin(), vertices_.end(), vertex) == vertices_.end());

        IdxVector vs = vertices_;
        vs.push_back(vertex);
        return Simplex(new_id, vs);
    }

    Simplex join(Int vertex)
    {
        return join(k_invalid_id, vertex);
    }

    bool operator==(const Simplex& other) const
    {
        // ignore id_ ?
        return get_id() == other.get_id() and vertices_ == other.vertices_;
    }

    bool operator!=(const Simplex& other) const
    {
        return !(*this == other);
    }

    template<typename I>
    friend std::ostream& operator<<(std::ostream&, const Simplex<I>&);

    Uid get_uid() const { return uid_; }

    std::string repr() const
    {
        std::stringstream out;
        out << "Simplex(id_=" << id_ << ", vertices_=[";

        for(size_t i = 0 ; i < vertices_.size() - 1 ; ++i)
            out << vertices_[i] << ", ";

        out << vertices_[vertices_.size() - 1] << "])";

        return out.str();
    }

    using UidHasher = std::hash<Uid>;
    using UidSet = std::unordered_set<Uid, UidHasher>;
};

template<typename I>
std::ostream& operator<<(std::ostream& out, const Simplex<I>& s)
{
    out << "[";

    for(size_t i = 0 ; i < s.vertices_.size() - 1 ; ++i)
        out << s.vertices_[i] << ", ";

    out << s.vertices_[s.vertices_.size() - 1] << "]";

    return out;
}
}

namespace std {

// libc++ provides std::hash for __int128 as a non-standard extension; libstdc++
// does not. Without these specializations, Simplex<Int>::UidHasher = std::hash<Uid>
// would not compile on gcc.
#if defined(__GLIBCXX__)
template<>
struct hash<unsigned __int128> {
    std::size_t operator()(unsigned __int128 v) const noexcept
    {
        auto lo = static_cast<std::uint64_t>(v);
        auto hi = static_cast<std::uint64_t>(v >> 64);
        std::size_t seed = std::hash<std::uint64_t>{}(lo);
        oineus::hash_combine(seed, hi);
        return seed;
    }
};

template<>
struct hash<__int128> {
    std::size_t operator()(__int128 v) const noexcept
    {
        return std::hash<unsigned __int128>{}(static_cast<unsigned __int128>(v));
    }
};
#endif

template<class Int>
struct hash<oineus::VREdge<Int>> {
    std::size_t operator()(const oineus::VREdge<Int>& p) const
    {
        std::size_t seed = 0;
        oineus::hash_combine(seed, p.x);
        oineus::hash_combine(seed, p.y);
        return seed;
    }
};

template<class Int>
struct hash<oineus::Simplex<Int>> {
    std::size_t operator()(const oineus::Simplex<Int>& sigma) const
    {
        std::size_t seed = 0;
        oineus::hash_combine(seed, sigma.get_id());
        oineus::hash_combine(seed, sigma.get_uid());
        return seed;
    }
};
} // namespace std
