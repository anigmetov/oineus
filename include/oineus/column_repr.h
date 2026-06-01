#pragma once
#ifndef OINEUS_COLUMN_REPR_H
#define OINEUS_COLUMN_REPR_H

#include <cstdint>
#include <vector>
#include <set>
#include <algorithm>
#include <ostream>
#include <atomic>

#include "sparse_matrix.h"

// Pluggable "working column" data structures for boundary-matrix reduction,
// following the accelerated column representations of the PHAT paper
// (Bauer, Kerber, Reininghaus, Wagner, J. Symbolic Computation 2017).
//
// The at-rest storage of every column stays a sorted std::vector<Int>; only the
// transient residual column used during reduction varies. Each working column
// type below offers the same small interface; GenericSparseMatrixTraits /
// GenericRVMatrixTraits adapt them to the trait API consumed by the reduction
// kernels in decomposition.h.
//
// Interface a working column must provide (Int = signed index type):
//   void reserve(size_t n_rows);                 // size dense buffers (no-op for sparse)
//   void load(const std::vector<Int>& sorted);   // clear + fill from a sorted column
//   void clear();                                 // reset to empty, O(nonzeros)
//   bool is_zero() const;
//   Int  low() const;                             // max set index, -1 if empty
//   void add(const std::vector<Int>& pivot);      // XOR a sorted pivot column
//   void to_vector(std::vector<Int>& out) const;  // sorted dump (non-destructive)
//   size_t size() const;                          // logical popcount (stats only)

namespace oineus {

// ---------------------------------------------------------------------------
// SetColumn: std::set<Int>. Reproduces the current Oineus baseline (PHAT A-Set).
// ---------------------------------------------------------------------------
template<class Int>
struct SetColumn {
    std::set<Int> data_;

    void reserve(size_t) { }
    void clear() { data_.clear(); }
    void load(const std::vector<Int>& col) { data_.clear(); data_.insert(col.begin(), col.end()); }

    void add(const std::vector<Int>& pivot)
    {
        for (Int e : pivot) {
            auto res = data_.insert(e);
            if (!res.second)
                data_.erase(res.first);
        }
    }

    bool is_zero() const { return data_.empty(); }
    Int low() const { return data_.empty() ? Int(-1) : *data_.rbegin(); }
    void to_vector(std::vector<Int>& out) const { out.assign(data_.begin(), data_.end()); }
    size_t size() const { return data_.size(); }
};

// ---------------------------------------------------------------------------
// HeapColumn: std::vector<Int> as a binary max-heap with lazy XOR
// (PHAT A-Heap / heap_pivot_column). Duplicates accumulate and cancel in pairs
// when the pivot is queried.
// ---------------------------------------------------------------------------
template<class Int>
struct HeapColumn {
    // mutable: low()/is_zero()/to_vector() prune the lazy heap to expose the pivot
    mutable std::vector<Int> heap_;

    void reserve(size_t) { }
    void clear() { heap_.clear(); }

    void load(const std::vector<Int>& col)
    {
        heap_.assign(col.begin(), col.end());
        std::make_heap(heap_.begin(), heap_.end());
    }

    void add(const std::vector<Int>& pivot)
    {
        for (Int e : pivot) {
            heap_.push_back(e);
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    // Remove and return the max index with odd multiplicity, canceling pairs.
    Int pop_pivot() const
    {
        if (heap_.empty())
            return Int(-1);
        Int piv = heap_.front();
        std::pop_heap(heap_.begin(), heap_.end());
        heap_.pop_back();
        while (!heap_.empty() && heap_.front() == piv) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.pop_back();
            if (heap_.empty()) { piv = Int(-1); break; }
            piv = heap_.front();
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.pop_back();
        }
        return piv;
    }

    Int low() const
    {
        Int piv = pop_pivot();
        if (piv != Int(-1)) {
            heap_.push_back(piv);
            std::push_heap(heap_.begin(), heap_.end());
        }
        return piv;
    }

    bool is_zero() const { return low() == Int(-1); }

    void to_vector(std::vector<Int>& out) const
    {
        out.clear();
        Int piv;
        while ((piv = pop_pivot()) != Int(-1))
            out.push_back(piv);
        std::reverse(out.begin(), out.end());
        // restore the heap to its (pruned) logical content -- keep this non-destructive
        heap_.assign(out.begin(), out.end());
        std::make_heap(heap_.begin(), heap_.end());
    }

    size_t size() const { return heap_.size(); }
};

// ---------------------------------------------------------------------------
// FullColumn: dense bit vector + lazy max-heap of candidate indices
// (PHAT A-Full / full_pivot_column). The bitset is authoritative; the heap only
// accelerates the pivot query.
// ---------------------------------------------------------------------------
template<class Int>
struct FullColumn {
    std::vector<uint64_t> bits_;
    mutable std::vector<Int> heap_;    // candidate pivots (may be stale / duplicated)
    std::vector<size_t> dirty_;        // level words touched since clear
    std::vector<uint8_t> in_dirty_;
    size_t nnz_ {0};

    void reserve(size_t n)
    {
        size_t nw = (n + 63) / 64;
        if (nw > bits_.size()) {
            bits_.assign(nw, 0);
            in_dirty_.assign(nw, 0);
        }
    }

    void touch(size_t w)
    {
        if (!in_dirty_[w]) { in_dirty_[w] = 1; dirty_.push_back(w); }
    }

    void clear()
    {
        for (size_t w : dirty_) { bits_[w] = 0; in_dirty_[w] = 0; }
        dirty_.clear();
        heap_.clear();
        nnz_ = 0;
    }

    bool test(Int i) const { return (bits_[size_t(i) >> 6] >> (i & 63)) & 1ULL; }

    void flip(Int i)
    {
        size_t w = size_t(i) >> 6;
        uint64_t m = 1ULL << (i & 63);
        bool was_set = bits_[w] & m;
        bits_[w] ^= m;
        touch(w);
        if (was_set) {
            nnz_--;
        } else {
            nnz_++;
            heap_.push_back(i);
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    void load(const std::vector<Int>& col) { clear(); for (Int e : col) flip(e); }
    void add(const std::vector<Int>& pivot) { for (Int e : pivot) flip(e); }

    bool is_zero() const { return nnz_ == 0; }

    Int low() const
    {
        while (!heap_.empty()) {
            Int t = heap_.front();
            if (test(t))
                return t;
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.pop_back();
        }
        return Int(-1);
    }

    void to_vector(std::vector<Int>& out) const
    {
        out.clear();
        std::vector<size_t> words(dirty_.begin(), dirty_.end());
        std::sort(words.begin(), words.end());
        for (size_t w : words) {
            uint64_t x = bits_[w];
            while (x) {
                int b = __builtin_ctzll(x);
                out.push_back(Int(w * 64 + b));
                x &= x - 1;
            }
        }
    }

    size_t size() const { return nnz_; }
};

// ---------------------------------------------------------------------------
// BitTreeColumn: hierarchical 64-ary dense bitset (PHAT A-Bit-Tree /
// bit_tree_pivot_column). A summary bit at level L is set iff its 64-bit child
// word at level L-1 is nonzero, giving O(log_64 n) pivot/insert. Touched words
// are tracked so clear()/to_vector() cost O(nonzeros), not O(n).
// ---------------------------------------------------------------------------
template<class Int>
struct BitTreeColumn {
    std::vector<std::vector<uint64_t>> levels_;
    std::vector<size_t> dirty_;        // level-0 word indices touched since clear
    std::vector<uint8_t> in_dirty_;
    size_t nnz_ {0};
    size_t n_bits_ {0};

    void reserve(size_t n)
    {
        if (n <= n_bits_)
            return;
        n_bits_ = n;
        levels_.clear();
        size_t len = (n + 63) / 64;
        levels_.push_back(std::vector<uint64_t>(len, 0));
        while (len > 1) {
            len = (len + 63) / 64;
            levels_.push_back(std::vector<uint64_t>(len, 0));
        }
        in_dirty_.assign((n + 63) / 64, 0);
        dirty_.clear();
        nnz_ = 0;
    }

    void touch(size_t w)
    {
        if (!in_dirty_[w]) { in_dirty_[w] = 1; dirty_.push_back(w); }
    }

    // Set or clear summary bit `idx` at `level`, propagating up while a word's
    // zero-status flips.
    void propagate(size_t level, size_t idx, bool set)
    {
        while (level < levels_.size()) {
            size_t w = idx >> 6;
            uint64_t m = 1ULL << (idx & 63);
            bool was_zero = (levels_[level][w] == 0);
            if (set) levels_[level][w] |= m; else levels_[level][w] &= ~m;
            bool now_zero = (levels_[level][w] == 0);
            if (was_zero == now_zero)
                return;
            idx = w;
            set = !now_zero;
            ++level;
        }
    }

    void flip(Int i)
    {
        size_t w = size_t(i) >> 6;
        uint64_t m = 1ULL << (i & 63);
        uint64_t before = levels_[0][w];
        levels_[0][w] = before ^ m;
        if (before & m) nnz_--; else nnz_++;
        touch(w);
        bool was_zero = (before == 0);
        bool now_zero = (levels_[0][w] == 0);
        if (was_zero != now_zero)
            propagate(1, w, !now_zero);
    }

    void clear()
    {
        for (size_t w : dirty_) {
            if (levels_[0][w] != 0) {
                levels_[0][w] = 0;
                propagate(1, w, false);
            }
            in_dirty_[w] = 0;
        }
        dirty_.clear();
        nnz_ = 0;
    }

    void load(const std::vector<Int>& col) { clear(); for (Int e : col) flip(e); }
    void add(const std::vector<Int>& pivot) { for (Int e : pivot) flip(e); }

    bool is_zero() const { return nnz_ == 0; }

    Int low() const
    {
        if (nnz_ == 0)
            return Int(-1);
        int L = int(levels_.size()) - 1;
        uint64_t topword = levels_[L][0];
        size_t pos = 63 - __builtin_clzll(topword);
        for (int lev = L; lev > 0; --lev) {
            uint64_t word = levels_[lev - 1][pos];
            pos = pos * 64 + (63 - __builtin_clzll(word));
        }
        return Int(pos);
    }

    void to_vector(std::vector<Int>& out) const
    {
        out.clear();
        std::vector<size_t> words(dirty_.begin(), dirty_.end());
        std::sort(words.begin(), words.end());
        for (size_t w : words) {
            uint64_t x = levels_[0][w];
            while (x) {
                int b = __builtin_ctzll(x);
                out.push_back(Int(w * 64 + b));
                x &= x - 1;
            }
        }
    }

    size_t size() const { return nnz_; }
};

template<class Int>
std::ostream& operator<<(std::ostream& out, const SetColumn<Int>& c)
{
    std::vector<Int> v; c.to_vector(v);
    out << "SetColumn[";
    for (auto e : v) out << e << " ";
    out << "]";
    return out;
}

template<class Int>
std::ostream& operator<<(std::ostream& out, const HeapColumn<Int>& c)
{
    std::vector<Int> v; c.to_vector(v);
    out << "HeapColumn[";
    for (auto e : v) out << e << " ";
    out << "]";
    return out;
}

template<class Int>
std::ostream& operator<<(std::ostream& out, const FullColumn<Int>& c)
{
    std::vector<Int> v; c.to_vector(v);
    out << "FullColumn[";
    for (auto e : v) out << e << " ";
    out << "]";
    return out;
}

template<class Int>
std::ostream& operator<<(std::ostream& out, const BitTreeColumn<Int>& c)
{
    std::vector<Int> v; c.to_vector(v);
    out << "BitTreeColumn[";
    for (auto e : v) out << e << " ";
    out << "]";
    return out;
}

// ---------------------------------------------------------------------------
// Generic Z_2 trait, parameterized by the working column type. Mirrors
// SimpleSparseMatrixTraits<Int,2> but routes residual-column operations through
// WorkColT. Storage (Column) stays a sorted std::vector<Int>.
// ---------------------------------------------------------------------------
template<typename Int_, typename WorkColT>
struct GenericSparseMatrixTraits {
    using Int = Int_;
    using Entry = Int;

    using Column = std::vector<Entry>;
    using Matrix = std::vector<Column>;
    using PColumn = Column*;
    using APColumn = std::atomic<PColumn>;
    using AMatrix = std::vector<APColumn>;

    using CachedColumn = WorkColT;

    static bool is_zero(const Column* c) { return c->empty(); }
    static bool is_zero(const Column& c) { return c.empty(); }
    static bool is_zero(const CachedColumn& col) { return col.is_zero(); }

    static Int low(const Column* c) { return c->empty() ? Int(-1) : c->back(); }
    static Int low(const CachedColumn& col) { return col.low(); }

    static size_t r_column_size(const Column& col) { return col.size(); }
    static size_t v_column_size(const Column&) { return 0; }
    static size_t r_column_size(const CachedColumn& col) { return col.size(); }
    static size_t v_column_size(const CachedColumn&) { return 0; }

    static void reserve(CachedColumn& col, size_t n_rows) { col.reserve(n_rows); }

    static CachedColumn load_to_cache(const Column& col) { CachedColumn c; c.load(col); return c; }
    static CachedColumn load_to_cache(Column* col) { CachedColumn c; if (col) c.load(*col); return c; }

    // in-place variants for buffer reuse across columns
    static void load_to_cache(const Column& col, CachedColumn& c) { c.load(col); }
    static void load_to_cache(const Column* col, CachedColumn& c) { if (col) c.load(*col); else c.clear(); }

    static void add_to_cached(const Column& pivot, CachedColumn& reduced) { reduced.add(pivot); }
    static void add_to_cached(const Column* pivot, CachedColumn& reduced) { if (pivot) reduced.add(*pivot); }

    static PColumn load_from_cache(const CachedColumn& col)
    {
        if (col.is_zero())
            return nullptr;
        auto* c = new Column();
        col.to_vector(*c);
        return c;
    }

    static void load_from_cache(const CachedColumn& col, Column& out) { col.to_vector(out); }

    static void sort(Column& col) { std::sort(col.begin(), col.end()); }

    static void add_to_column(Column& a, const Column& b)
    {
        SimpleSparseMatrixTraits<Int, 2>::add_to_column(a, b);
    }

    static std::string check_col_duplicates(PColumn col)
    {
        return SimpleSparseMatrixTraits<Int, 2>::check_col_duplicates(col);
    }

    static Matrix eye(size_t n)
    {
        Matrix cols{n};
        for (Int i = 0; i < static_cast<Int>(n); ++i)
            cols[i].emplace_back(i);
        return cols;
    }
};

// ---------------------------------------------------------------------------
// Generic Z_2 RV trait: fused R + V working column (pair of WorkColT). Mirrors
// SimpleRVMatrixTraits<Int,2>. The R part needs low(); the V part never does.
// ---------------------------------------------------------------------------
template<typename Int_, typename WorkColT>
struct GenericRVMatrixTraits {
    using Int = Int_;
    using Entry = Int;

    using Column = RVColumn<Int, 2>;
    using PColumn = Column*;
    using APColumn = std::atomic<PColumn>;
    using AMatrix = std::vector<APColumn>;

    using CachedColumn = std::pair<WorkColT, WorkColT>;  // (R work, V work)

    static bool is_zero(const Column* col) { return col->is_zero(); }
    static bool is_zero(const CachedColumn& col) { return col.first.is_zero(); }

    static Int low(const Column* col) { return col->low(); }
    static Int low(const Column& col) { return col.low(); }
    static Int low(const CachedColumn& col) { return col.first.low(); }

    static size_t r_column_size(const Column& col) { return col.r_column.size(); }
    static size_t v_column_size(const Column& col) { return col.v_column.size(); }
    static size_t r_column_size(const CachedColumn& col) { return col.first.size(); }
    static size_t v_column_size(const CachedColumn& col) { return col.second.size(); }

    static void reserve(CachedColumn& col, size_t n_rows)
    {
        col.first.reserve(n_rows);
        col.second.reserve(n_rows);
    }

    static void add_to_cached(const Column& pivot, CachedColumn& reduced)
    {
        reduced.first.add(pivot.r_column);
        reduced.second.add(pivot.v_column);
    }

    static void add_to_cached(const Column* pivot, CachedColumn& reduced) { add_to_cached(*pivot, reduced); }

    static CachedColumn load_to_cache(const Column& col)
    {
        CachedColumn c;
        c.first.load(col.r_column);
        c.second.load(col.v_column);
        return c;
    }

    static CachedColumn load_to_cache(Column* col)
    {
        CachedColumn c;
        if (col) { c.first.load(col->r_column); c.second.load(col->v_column); }
        return c;
    }

    static void load_to_cache(const Column* col, CachedColumn& c)
    {
        if (col) { c.first.load(col->r_column); c.second.load(col->v_column); }
        else { c.first.clear(); c.second.clear(); }
    }

    static PColumn load_from_cache(const CachedColumn& col)
    {
        std::vector<Int> rvec, vvec;
        col.first.to_vector(rvec);
        col.second.to_vector(vvec);
        return new Column(std::move(rvec), std::move(vvec));
    }

    static std::string check_col_duplicates(PColumn col)
    {
        return SimpleRVMatrixTraits<Int, 2>::check_col_duplicates(col);
    }
};

} // namespace oineus

#endif // OINEUS_COLUMN_REPR_H
