#pragma once

#include <cstdint>
#include <ostream>
#include <vector>
#include <unordered_set>
#include <functional>
#include <utility>
#include <algorithm>
#include <cassert>

#include "common_defs.h"

namespace oineus {

// Shared geometry for a bit-packed simplicial filtration: just the field width.
// A Vietoris-Rips / alpha simplex is stored as its sorted vertex ids packed into a
// single machine word, each id in `bits` bits (bits = ceil(log2(n_points))). The
// width is the same for every cell, so the Filtration owns one PackedGeom and threads
// it into the cell's (co)boundary / materialization -- exactly like GridDomain for a
// cube or FrGeometry for a Freudenthal cell.
struct PackedGeom {
    int bits {0};
    bool operator==(const PackedGeom& o) const { return bits == o.bits; }
    bool operator!=(const PackedGeom& o) const { return not(*this == o); }
};

inline std::ostream& operator<<(std::ostream& out, const PackedGeom& g)
{
    out << "PackedGeom(bits=" << g.bits << ")";
    return out;
}

// Smallest field width to index n_points vertex ids 0..n_points-1 (ceil(log2),
// at least 1).
inline int packed_vertex_bits(size_t n_points)
{
    int b = 1;
    while ((size_t(1) << b) < n_points)
        ++b;
    return b;
}

// Does a (top_dim)-simplex over n_points vertices fit in Word when bit-packed?
// dim is stored on the cell (no dim bits in the word), so a d-simplex needs
// (d+1)*bits bits. The VR/alpha builder uses this to pick the smallest fitting word
// (uint64_t -> __int128 -> fall back to Fat).
template<class Word>
inline bool bit_packing_fits(size_t n_points, dim_type top_dim)
{
    int b = packed_vertex_bits(n_points);
    return (static_cast<size_t>(top_dim) + 1) * static_cast<size_t>(b) <= sizeof(Word) * 8;
}

// Bit-packed simplicial cell ENCODING: the sorted vertex ids packed low-to-high into
// `Word uid_` (field i = the i-th smallest vertex, in PackedGeom::bits bits), plus the
// stored dimension (the cell concept needs dim() without geometry; storing it also
// avoids any dim-from-packing ambiguity for the all-low-bits vertex 0). Used as the
// Enc of Simplex<Int, BitPacked<Int,Word>> for Vietoris-Rips / alpha, where vertex ids
// are arbitrary (sparse) -- so HasPackedBoundary is true (drop a field, repack) but
// UsesDenseUidIndex is false (the packed uid is wide -> hash) and HasDirectCoboundary
// is false (VR has no cheap direct coboundary -> antitranspose). The fat form is the
// same Simplex on the unpacked vertices (vertices(geom)).
template<class Int_, class Word_>
struct BitPacked {
    using Int = Int_;
    using Word = Word_;
    using Uid = Word;
    using Geometry = PackedGeom;
    using Boundary = std::vector<Word>;
    // materialized fat vertex ids (see vertices(geom)); the wrapper exposes this as its
    // IdxVector typedef, though a slim cell stores no vertices
    using IdxVector = std::vector<Int>;
    using UidHasher = std::hash<Word>;
    using UidSet = std::unordered_set<Uid, UidHasher>;

    Word uid_ {0};
    dim_type dim_ {0};

    BitPacked() = default;
    BitPacked(const BitPacked&) = default;
    BitPacked(BitPacked&&) noexcept = default;
    BitPacked& operator=(const BitPacked&) = default;
    BitPacked& operator=(BitPacked&&) noexcept = default;

    // direct construction from an already-packed word + dim
    BitPacked(Word uid, dim_type d) : uid_(uid), dim_(d) {}

    // pack an ascending vertex list, `bits` per field (caller supplies bits from the
    // PackedGeom; the cell does not store it). Templated on the container so the VR
    // builder's jemalloc-allocated vertex buffer packs without an intermediate copy;
    // the trailing decltype constrains it to size()-having containers so it never
    // competes with the BitPacked(Word, dim_type) ctor above for integral arguments.
    template<class Vec, class = decltype(std::declval<const Vec&>().size())>
    BitPacked(const Vec& sorted_vertices, int bits)
            : uid_(pack(sorted_vertices, bits)),
              dim_(static_cast<dim_type>(sorted_vertices.size()) - 1)
    {
        assert(std::is_sorted(sorted_vertices.begin(), sorted_vertices.end()));
    }

    static Word field_mask(int bits) { return (static_cast<Word>(1) << bits) - 1; }

    template<class Vec>
    static Word pack(const Vec& sorted_vertices, int bits)
    {
        Word w = 0;
        for (size_t i = 0; i < sorted_vertices.size(); ++i)
            w |= static_cast<Word>(sorted_vertices[i]) << (i * static_cast<size_t>(bits));
        return w;
    }

    dim_type dim() const { return dim_; }

    Word get_uid() const { return uid_; }
    void set_uid() { throw std::runtime_error("Changing UID of a bit-packed cell is prohibited."); }

    // Buffer boundary: drop each vertex field in turn and repack the remaining d fields
    // contiguously (they stay ascending), emitting each facet's packed word. No
    // intermediate vector -- this is the alloc-elided body the Filtration builders call.
    template<typename Visitor>
    void boundary_into(const Geometry& g, Visitor&& visit) const
    {
        if (dim_ == 0)
            return;
        const int b = g.bits;
        const Word mask = field_mask(b);
        for (dim_type drop = 0; drop <= dim_; ++drop) {
            Word fw = 0;
            dim_type pos = 0;
            for (dim_type i = 0; i <= dim_; ++i) {
                if (i == drop)
                    continue;
                Word field = (uid_ >> (i * static_cast<size_t>(b))) & mask;
                fw |= field << (pos * static_cast<size_t>(b));
                ++pos;
            }
            visit(fw);
        }
    }

    std::vector<Word> boundary(const Geometry& g) const
    {
        std::vector<Word> result;
        boundary_into(g, [&result](Word u) { result.push_back(u); });
        return result;
    }

    // The grid vertex ids of this cell (slim->fat materialization): unpack the d+1
    // fields. They come out ascending (field 0 is the smallest), the canonical fat
    // Simplex vertex order. The fat Simplex on these vertices is the honest cell this
    // packed form stands in for.
    std::vector<Int> vertices(const Geometry& g) const
    {
        const int b = g.bits;
        const Word mask = field_mask(b);
        std::vector<Int> vs;
        vs.reserve(dim_ + 1);
        for (dim_type i = 0; i <= dim_; ++i)
            vs.push_back(static_cast<Int>((uid_ >> (i * static_cast<size_t>(b))) & mask));
        return vs;
    }

    bool operator==(const BitPacked& other) const { return uid_ == other.uid_ and dim_ == other.dim_; }
    bool operator!=(const BitPacked& other) const { return not(*this == other); }
};

template<class Int, class Word>
std::ostream& operator<<(std::ostream& out, const BitPacked<Int, Word>& c)
{
    out << "BitPacked(uid=" << c.uid_ << ", dim=" << c.dim_ << ")";
    return out;
}

// Bit-packed VR/alpha simplices: a packed buffer boundary (drop a field, repack), but
// a sparse/wide uid (-> hash index, not a flat direct-address one) and no cheap direct
// coboundary (-> antitranspose). The wrapper's traits (simplex.h) delegate to these.
template<class Int, class Word>
struct HasPackedBoundary<BitPacked<Int, Word>> : std::true_type {};
template<class Int, class Word>
struct HasDirectCoboundary<BitPacked<Int, Word>> : std::false_type {};
template<class Int, class Word>
struct UsesDenseUidIndex<BitPacked<Int, Word>> : std::false_type {};

} // namespace oineus
