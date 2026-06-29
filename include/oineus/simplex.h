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
#include <type_traits>

#include "common_defs.h"

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
    for(IntIn i = 1; i <= k; i++) {
        result *= (n - i + 1);
        assert(result % static_cast<IntOut>(i) == 0);
        result /= static_cast<IntOut>(i);
    }
    return result;
}

// True iff binomial(n, k) <= bound, evaluated WITHOUT 128-bit overflow. The combinatorial
// number system decode (vertices_from_simplex_uid) only needs this comparison, never the
// possibly-huge exact value, so we build C(n,i) iteratively and bail out the moment a
// multiply would overflow: when C(n,i-1) * (n-i+1) exceeds the Uid range, C(n,i) >= 2^124
// (for the small k we use), which already exceeds any 124-bit bound, so C(n,k) > bound.
template<typename Uid>
bool comb_leq(Uid n, Uid k, Uid bound)
{
    if (n < k)
        return true;            // C(n, k) == 0
    Uid result = 1;             // C(n, 0)
    for(Uid i = 1; i <= k; ++i) {
        Uid factor = n - i + 1;
        if (result > std::numeric_limits<Uid>::max() / factor)
            return false;       // C(n, i) (hence C(n, k)) overflows 124-bit bound
        result = result * factor / i;
    }
    return result <= bound;
}

// combinatorial simplex numbering, as in Ripser (see Bauer's paper).
// Encodes dimension info in the 4 most significant bits of the 128-bit result.
template<typename IntIn, typename Alloc = std::allocator<IntIn>>
unsigned __int128 simplex_uid(const std::vector<IntIn, Alloc>& vertices)
{
    using Uid = unsigned __int128;
    Uid dim_info = static_cast<Uid>(vertices.size() + 1) << (8 * sizeof(Uid) - 4);
    Uid uid = 0;
    for(IntIn i = 0; i < static_cast<IntIn>(vertices.size()); i++) {
        uid += comb<IntIn, Uid>(vertices[i], i + 1);
    }
    return uid | dim_info;
}

// Inverse of simplex_uid: recover the ascending vertex ids encoded in a combinatorial
// uid. The top 4 bits hold (n_vertices + 1); the low 124 bits hold
// sum_i C(vertices[i], i+1) with vertices ascending (the combinatorial number system).
// We peel terms from the highest position k = n_vertices down to 1: the largest c with
// comb(c, k) <= remainder is vertices[k-1]. comb(.,k) is monotone in its first argument,
// so each c is found by an exponential probe followed by a binary search -- O(d log V)
// comb evaluations, no precomputed table. This is the Python-facing translation that maps
// a (fat) simplex's universal combinatorial uid back to its vertex set, so that the slim /
// bit-packed encodings can re-key it into their own internal uid for uid-based lookups.
template<typename Int>
std::vector<Int> vertices_from_simplex_uid(unsigned __int128 uid)
{
    using Uid = unsigned __int128;
    constexpr unsigned dim_shift = 8 * sizeof(Uid) - 4;
    Int n_vertices = static_cast<Int>(uid >> dim_shift) - 1;
    if (n_vertices <= 0)
        throw std::runtime_error("vertices_from_simplex_uid: malformed uid (no vertices)");

    Uid rem = uid & ((static_cast<Uid>(1) << dim_shift) - 1);
    const Uid int_max = static_cast<Uid>(std::numeric_limits<Int>::max());
    std::vector<Int> vertices(n_vertices);

    for(Int k = n_vertices; k >= 1; --k) {
        Uid kk = static_cast<Uid>(k);
        // a vertex id beyond the Int range is not a cell of any (Int-indexed) filtration;
        // if even comb(int_max, k) still fits under rem the decoded vertex overflows Int,
        // so report the uid as out of range (the binding layer maps this to "not present",
        // matching the fat accessor, instead of returning a truncated/garbage vertex)
        if (comb_leq<Uid>(int_max, kk, rem))
            throw std::out_of_range("vertices_from_simplex_uid: uid out of range");
        // largest c in [k-1, int_max] with comb(c, k) <= rem (comb is monotone in c, and
        // comb_leq(int_max, ...) is false here so the boundary is bracketed); binary search
        Uid lo = kk - 1, hi = int_max;          // comb(k-1, k) == 0 <= rem is the floor
        while (hi - lo > 1) {
            Uid mid = lo + (hi - lo) / 2;
            if (comb_leq<Uid>(mid, kk, rem))
                lo = mid;
            else
                hi = mid;
        }
        vertices[k - 1] = static_cast<Int>(lo);
        rem -= comb<Uid, Uid>(lo, kk);          // exact: comb(lo, k) <= rem < 2^124, no overflow
    }

    assert(std::is_sorted(vertices.begin(), vertices.end()));
    return vertices;
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

// A simplicial cell is Simplex<Int, Enc>: a thin wrapper that carries the user id
// and delegates storage + (co)boundary to an encoding policy Enc. The default Fat
// encoding (below) stores the explicit, sorted vertex list -- the universal fat
// form used by user-defined, Vietoris-Rips and alpha filtrations -- so that
// Simplex<Int> means Simplex<Int, Fat<Int>> and the historical behavior is
// preserved exactly. Other encodings (bit-packed VR ids, Freudenthal anchor+type)
// are added as further Enc policies; each materializes to the same fat vertex list
// on access. This (Stage C of the slim-cell refactor) is the no-behavior-change
// wrapper introduction with Enc = Fat.

// Fat encoding: an explicit sorted vertex list plus the cached combinatorial uid.
// Self-contained -- its (co)boundary needs no shared geometry, so Geometry =
// NoGeometry and it provides a no-argument boundary().
template<typename Int_>
struct Fat {
    using Int = Int_;
    // Vertex storage routed through jemalloc when the build links it (see
    // JeAllocator in common_defs.h); identical type/ABI when it does not, so the
    // OINEUS_USE_JEMALLOC flag only changes the malloc backend, not the type.
    using IdxVector = std::vector<Int, JeAllocator<Int>>;

    using Uid = unsigned __int128;
    // for Z2 only for now
    using Boundary = std::vector<Uid>;
    using Geometry = NoGeometry;
    using UidHasher = std::hash<Uid>;
    using UidSet = std::unordered_set<Uid, UidHasher>;

    Uid uid_ {};
    IdxVector vertices_;

    Fat() = default;
    Fat(const Fat&) = default;
    Fat(Fat&&) noexcept = default;
    Fat& operator=(const Fat&) = default;
    Fat& operator=(Fat&&) noexcept = default;

    explicit Fat(const IdxVector& _vertices)
            :vertices_(_vertices)
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        if (vertices_.size() > 1)
            std::sort(vertices_.begin(), vertices_.end());

        set_uid();
    }

    // Caller promises _vertices is already in ascending order, so we skip the
    // std::sort step. Used by the in-order (VRE) Vietoris-Rips construction, which
    // builds vertex lists pre-sorted by construction (see generate_cofacets in
    // vietoris_rips_inorder.h). Takes by rvalue reference because the typical
    // caller has already moved its working IdxVector into the call.
    Fat(presorted_t, IdxVector&& _vertices)
            :vertices_(std::move(_vertices))
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        assert(std::is_sorted(vertices_.begin(), vertices_.end()));

        set_uid();
    }

    dim_type dim() const { return static_cast<Int>(vertices_.size()) - 1; }

    Uid get_uid() const { return uid_; }
    // uids are set in parallel
    void set_uid() { uid_ = simplex_uid<Int>(vertices_); }

    const IdxVector& get_vertices() const { return vertices_; }

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

    bool operator==(const Fat& other) const { return vertices_ == other.vertices_; }
    bool operator!=(const Fat& other) const { return !(*this == other); }
};

template<typename Int>
std::ostream& operator<<(std::ostream& out, const Fat<Int>& f)
{
    out << "[";

    for(size_t i = 0 ; i + 1 < f.vertices_.size() ; ++i)
        out << f.vertices_[i] << ", ";

    out << f.vertices_[f.vertices_.size() - 1] << "]";

    return out;
}

template<typename Int_, typename Enc_ = Fat<Int_>>
struct Simplex {
    using Int = Int_;
    using Enc = Enc_;

    // The cell-concept typedefs come from the encoding; for Fat these are the
    // historical Simplex types (IdxVector / unsigned __int128 uid / NoGeometry).
    using IdxVector = typename Enc::IdxVector;
    using Uid = typename Enc::Uid;
    using Boundary = typename Enc::Boundary;
    using Geometry = typename Enc::Geometry;
    using UidHasher = typename Enc::UidHasher;
    using UidSet = typename Enc::UidSet;

    static constexpr Int k_invalid_id = Int(-1);

    Int id_ {k_invalid_id};
    Enc enc_;

    Simplex() = default;
    Simplex(const Simplex&) = default;
    Simplex(Simplex&&) noexcept = default;
    Simplex& operator=(const Simplex&) = default;
    Simplex& operator=(Simplex&&) noexcept = default;

    Simplex(const IdxVector& _vertices)
            :enc_(_vertices)
    {
        // a vertex (0-simplex) takes its single vertex as its user id, matching
        // the historical Simplex(vertices) behavior; higher-dimensional cells keep
        // k_invalid_id until the filtration assigns one.
        if (enc_.dim() == 0)
            id_ = enc_.get_vertices()[0];
    }

    Simplex(presorted_t, IdxVector&& _vertices)
            :enc_(presorted, std::move(_vertices))
    {
        if (enc_.dim() == 0)
            id_ = enc_.get_vertices()[0];
    }

    Simplex(const Int _id, const IdxVector& _vertices)
            :id_(_id), enc_(_vertices)
    {
    }

    // Construct directly from an already-built encoding -- used by packed encodings
    // (Freudenthal anchor+type, bit-packed) whose payload is a uid + dim rather than a
    // vertex list. The user id defaults to invalid until the filtration assigns one.
    explicit Simplex(Enc _enc, Int _id = k_invalid_id)
            :id_(_id), enc_(std::move(_enc))
    {
    }

    dim_type dim() const { return enc_.dim(); }

    Int get_id() const { return id_; }
    void set_id(Int new_id) { id_ = new_id; }

    Uid get_uid() const { return enc_.get_uid(); }
    void set_uid() { enc_.set_uid(); }

    // get_vertices / no-arg boundary / join / repr are valid only for self-contained
    // (NoGeometry) encodings, which carry their own vertex list -- i.e. Fat. They are
    // SFINAE-gated member templates (the gate depends on the function's own template
    // parameter E) so they simply do NOT exist for geometry-bearing encodings
    // (Freudenthal anchor+type, bit-packed): a stray slim/packed call is a clean
    // "no such method", and trait detection (e.g. CellWithValue::get_vertices /
    // boundary, which decltype-probe these) is honest. Geometry-bearing encodings use
    // boundary_into / coboundary_into / vertices(geom) (below) instead. Being templates,
    // they are bound via lambdas, not &Simplex::method member pointers.
    template<class E = Enc, std::enable_if_t<std::is_same_v<typename E::Geometry, NoGeometry>, int> = 0>
    const IdxVector& get_vertices() const { return enc_.get_vertices(); }

    template<class E = Enc, std::enable_if_t<std::is_same_v<typename E::Geometry, NoGeometry>, int> = 0>
    Boundary boundary() const { return enc_.boundary(); }

    // Geometry-bearing encodings (Freudenthal anchor+type, bit-packed) expose
    // alloc-elided buffer (co)boundary and on-the-fly vertex materialization that need
    // the shared geometry; these forward to the encoding. They are member templates,
    // so each is instantiated only for an encoding that actually provides it -- the
    // Filtration's packed builders call boundary_into / coboundary_into, and the
    // fat-cell materialization (Fattener / tests) calls vertices.
    template<class Visitor>
    void boundary_into(const Geometry& geom, Visitor&& visit) const
    {
        enc_.boundary_into(geom, std::forward<Visitor>(visit));
    }

    template<class Visitor>
    void coboundary_into(const Geometry& geom, Visitor&& visit) const
    {
        enc_.coboundary_into(geom, std::forward<Visitor>(visit));
    }

    template<class E = Enc>
    auto vertices(const Geometry& geom) const -> decltype(std::declval<const E&>().vertices(geom))
    {
        return enc_.vertices(geom);
    }

    // create a new simplex by joining with vertex and assign id to it (Fat-only, see above)
    template<class E = Enc, std::enable_if_t<std::is_same_v<typename E::Geometry, NoGeometry>, int> = 0>
    Simplex join(Int new_id, Int vertex) const
    {
        const IdxVector& vs = enc_.get_vertices();
        // new vertex must not be present in this->vertices
        assert(std::find(vs.begin(), vs.end(), vertex) == vs.end());

        IdxVector new_vertices = vs;
        new_vertices.push_back(vertex);
        return Simplex(new_id, new_vertices);
    }

    template<class E = Enc, std::enable_if_t<std::is_same_v<typename E::Geometry, NoGeometry>, int> = 0>
    Simplex join(Int vertex)
    {
        return join(k_invalid_id, vertex);
    }

    bool operator==(const Simplex& other) const
    {
        // ignore id_ ?
        return get_id() == other.get_id() and enc_ == other.enc_;
    }

    bool operator!=(const Simplex& other) const
    {
        return !(*this == other);
    }

    template<typename I, typename E>
    friend std::ostream& operator<<(std::ostream&, const Simplex<I, E>&);

    template<class E = Enc, std::enable_if_t<std::is_same_v<typename E::Geometry, NoGeometry>, int> = 0>
    std::string repr() const
    {
        std::stringstream out;
        const IdxVector& vs = enc_.get_vertices();
        out << "Simplex(id_=" << id_ << ", vertices_=[";

        for(size_t i = 0 ; i + 1 < vs.size() ; ++i)
            out << vs[i] << ", ";

        out << vs[vs.size() - 1] << "])";

        return out.str();
    }
};

template<typename I, typename E>
std::ostream& operator<<(std::ostream& out, const Simplex<I, E>& s)
{
    out << s.enc_;
    return out;
}

// A Simplex's cell-policy traits delegate to its encoding's: a Simplex is
// packed-boundary / direct-coboundary / dense-uid-indexed exactly when its encoding
// is. Fat leaves all three at the default false (the historical fat Simplex); packed
// encodings (Freudenthal anchor+type, bit-packed) specialize their own.
template<class Int, class Enc> struct HasPackedBoundary<Simplex<Int, Enc>> : HasPackedBoundary<Enc> {};
template<class Int, class Enc> struct HasDirectCoboundary<Simplex<Int, Enc>> : HasDirectCoboundary<Enc> {};
template<class Int, class Enc> struct UsesDenseUidIndex<Simplex<Int, Enc>> : UsesDenseUidIndex<Enc> {};
}

namespace std {

// Simplex<Int>::UidHasher = std::hash<Uid> needs std::hash for __int128 when
// Uid is __int128 (packed encodings). libc++ provides it as a non-standard
// extension. libstdc++ provides it in two cases, and defining ours on top then
// collides (redefinition error):
//   - GNU mode (-std=gnu++NN): under __GLIBCXX_TYPE_INT_N_0
//   - strict mode (-std=c++NN) since GCC 16 (_GLIBCXX_RELEASE >= 16): under
//     __STRICT_ANSI__ && __SIZEOF_INT128__ (bits/functional_hash.h)
// Older libstdc++ provides neither in strict mode, which is what we build with;
// supply our own only when libstdc++ provides neither.
#if defined(__GLIBCXX__) \
 && !defined(__GLIBCXX_TYPE_INT_N_0) \
 && !(defined(__STRICT_ANSI__) && defined(__SIZEOF_INT128__) && _GLIBCXX_RELEASE >= 16)
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

template<class Int, class Enc>
struct hash<oineus::Simplex<Int, Enc>> {
    std::size_t operator()(const oineus::Simplex<Int, Enc>& sigma) const
    {
        std::size_t seed = 0;
        oineus::hash_combine(seed, sigma.get_id());
        oineus::hash_combine(seed, sigma.get_uid());
        return seed;
    }
};
} // namespace std
