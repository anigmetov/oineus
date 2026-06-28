#pragma once
#ifndef OINEUS_SPARSE_MATRIX_H
#define OINEUS_SPARSE_MATRIX_H

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <set>
#include <unordered_set>
#include <cassert>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <memory>
#include <boost/container/small_vector.hpp>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

#include "common_defs.h"   // JeAllocator + guarded <jemalloc/jemalloc.h>
#include "profile.h"

namespace oineus {

#ifndef OINEUS_COL_INLINE_CAP
#define OINEUS_COL_INLINE_CAP 4
#endif

// JeAllocator (jemalloc-routed allocator) now lives in common_defs.h so it can
// be shared by the simplex/cell vertex vectors as well as the reduction columns.

#ifdef OINEUS_COL_USE_STD_VECTOR
// Benchmark baseline: plain std::vector columns -- no SBO, no jemalloc routing.
template<class Int>
using SparseColumn = std::vector<Int>;
#else
// At-rest / working sparse column. Small-buffer-optimized vector whose OVERFLOW
// buffer is routed to jemalloc via JeAllocator, and whose heap-allocated OBJECT
// (a column `new`'d as a parallel-reduction working column) is routed to jemalloc
// via the class-scoped operator new/delete. Deriving from small_vector keeps the
// full container interface; `delete p` auto-dispatches to operator delete, so the
// reduction's `new`/`delete` sites need no changes.
template<class Int>
struct OinColumn : boost::container::small_vector<Int, OINEUS_COL_INLINE_CAP, JeAllocator<Int>> {
    using Base = boost::container::small_vector<Int, OINEUS_COL_INLINE_CAP, JeAllocator<Int>>;
    using Base::Base;
#ifdef OINEUS_USE_JEMALLOC
    static void* operator new(std::size_t n)
    {
        void* p = je_malloc(n);
        if (p == nullptr)
            throw std::bad_alloc();
        return p;
    }
    static void operator delete(void* p) noexcept { je_free(p); }
#endif
};

template<class Int>
using SparseColumn = OinColumn<Int>;
#endif

// Catch accidental N blowup of the inline buffer. With N=4, the column is 56 bytes
// here (small_vector header + the 4-element 8-byte inline buffer); 64 keeps a margin
// while still flagging an order-of-magnitude mistake. JeAllocator is stateless, so it
// does not change the size vs the default allocator.
static_assert(sizeof(SparseColumn<long int>) <= 64,
        "SparseColumn inline buffer unexpectedly large -- check OINEUS_COL_INLINE_CAP");

// Column-content predicates, templated on the container so they accept both
// the at-rest SBO column (SparseColumn) and a plain std::vector column that a
// caller might still hold. Only .empty()/.back() are used.
template<class Col>
bool is_zero(const Col& col)
{
    return col.empty();
}

template<class Col>
bool is_zero(const Col* col)
{
    return col->empty();
}

template<class Col>
auto low(const Col* col) -> typename Col::value_type
{
    return is_zero(col) ? typename Col::value_type(-1) : col->back();
}

template<class Col>
auto low(const Col& col) -> typename Col::value_type
{
    return is_zero(col) ? typename Col::value_type(-1) : col.back();
}

template<typename Int_, int P>
struct SimpleSparseMatrixTraits {
    using Int = Int_;
    using FieldElement = short;
    using Entry = std::pair<Int, FieldElement>;

    using Column = std::vector<Entry>;
    using Matrix = std::vector<Column>;
    using PColumn = Column*;
    using APColumn = std::atomic<PColumn>;
    using AMatrix = std::vector<APColumn>;

    static bool is_zero(const Column* c) { return c->empty(); }
    static bool is_zero(const Column& c) { return c.empty(); }

    static Int low(const Column* c)
    {
        return c->empty() ? -1 : c->back().first;
    }

    static size_t r_column_size(const Column& col) { return col.size(); }
    static size_t v_column_size([[maybe_unused]] const Column& col) { return 0; }

    static void add_column(const Column* col_a, const Column* col_b, Column* sum);

    static Column& r_data(Column* col) { return *col; }

    // Print a single column as "(row1, coeff1) (row2, coeff2) ..."
    // For the Z_p (P != 2) form, both row and coefficient are visible
    // because the column entries are (row, coefficient) pairs.
    static void print_column(std::ostream& out, const Column& col)
    {
        for (const auto& e : col)
            out << "(" << e.first << ", " << e.second << ") ";
    }

    // get identity matrix
    static Matrix eye(size_t n)
    {
        Matrix cols{n};

        for(size_t i = 0 ; i < n ; ++i)
            cols[i].emplace_back(i, 1);
        return cols;
    }
};

template<typename Int_>
struct SimpleSparseMatrixTraits<Int_, 2> {
    using Int = Int_;
    using Entry = Int;

    using Column = SparseColumn<Entry>;
    using Matrix = std::vector<Column>;
    using PColumn = Column*;
    using APColumn = std::atomic<PColumn>;
    using AMatrix = std::vector<APColumn>;

    using CachedColumn = std::set<Entry>;

    static bool is_zero(const Column* c) { return c->empty(); }
    static bool is_zero(const Column& c) { return c.empty(); }

    static size_t r_column_size(const Column& col) { return col.size(); }
    static size_t v_column_size([[maybe_unused]] const Column& col) { return 0; }

    static size_t r_column_size(const CachedColumn& col) { return col.size(); }
    static size_t v_column_size([[maybe_unused]] const CachedColumn& col) { return 0; }

    static Int low(const Column* c)
    {
        return c->empty() ? -1 : c->back();
    }

    static CachedColumn load_to_cache(const Column& col)
    {
        return CachedColumn(col.begin(), col.end());
    }

    static CachedColumn load_to_cache(Column* col)
    {
        if (col == nullptr)
            return CachedColumn();
        else
            return load_to_cache(*col);
    }

    static void add_to_cached(const Column& pivot, CachedColumn& reduced)
    {
        add_to_cached(&pivot, reduced);
    }

    static CachedColumn cached_identity_column(Int col_idx)
    {
        return CachedColumn({col_idx});
    }

    // sort by index
    static void sort(Column& col)
    {
        std::sort(col.begin(), col.end());
    }

    static void add_to_cached(const Column* pivot, CachedColumn& reduced)
    {
        for(auto e : *pivot) {
            auto iter_exists = reduced.insert(e);
            if (!iter_exists.second)
                reduced.erase(iter_exists.first);
        }
    }

    static Column& r_data(Column* col) { return *col; }

    // Print a single Z_2 column as space-separated row indices.
    // Coefficients are implicitly 1; nothing else to show.
    static void print_column(std::ostream& out, const Column& col)
    {
        for (const auto& e : col)
            out << e << " ";
    }

    static bool is_zero(const CachedColumn& col)
    {
        return col.empty();
    }

    static Int low(const CachedColumn& col)
    {
        return is_zero(col) ? -1 : *col.rbegin();
    }

    // Symmetric of low: smallest entry in the cached residual.
    // Used by row-form U inversion (compute_u_row_bounded), where the
    // residual is consumed top-down (V^T is lower unit-triangular ->
    // forward substitution -> top of residual strictly increases each
    // iteration).
    static Int top(const CachedColumn& col)
    {
        return is_zero(col) ? -1 : *col.begin();
    }

    static PColumn load_from_cache(const CachedColumn& col)
    {
        if (col.empty())
            return nullptr;
        else
            return new Column(col.begin(), col.end());
    }

    static void load_from_cache(const CachedColumn& cached_col, Column& col)
    {
        col = Column(cached_col.begin(), cached_col.end());
    }

    // get identity matrix
    static Matrix eye(size_t n)
    {
        Matrix cols{n};

        for(Int i = 0 ; i < static_cast<Int>(n); ++i)
            cols[i].emplace_back(i);
        return cols;
    }

    // return empty string, if there are no duplicates
    // otherwise return error message with column content
    static std::string check_col_duplicates(PColumn col)
    {
        if (col == nullptr) return "";
        std::set<Entry> seen;
        for(const Entry& e : *col) {
            if (seen.find(e) != seen.end()) {
                std::stringstream ss;
                ss << " ERROR DUPLICATES in " << (void*) col << ", [";
                for(auto x : *col)
                    ss << x << ", ";
                ss << "] ";
                return ss.str();
            }
            seen.insert(e);
        }
        return "";
    }

    // add col_b to col_a
    static void add_to_column(Column& col_a, const Column& col_b)
    {
        Column result;
        std::set_symmetric_difference(col_a.begin(), col_a.end(), col_b.begin(), col_b.end(), std::back_inserter(result));
        col_a.swap(result);
    }

    static Matrix col_to_row_format(const Matrix& col_format, size_t col_start=0,
                                    size_t col_end=std::numeric_limits<size_t>::max(),
                                    Int num_rows=-1) {
        if (col_format.empty()) {
            return {};
        }

        if (col_end > col_format.size()) {
            col_end = col_format.size();
        }

        // Determine the number of rows needed, if not given
        if (num_rows == -1) {
            for (const auto& col : col_format) {
                if (!col.empty()) {
                    num_rows = std::max(num_rows, col.back());
                }
            }
            num_rows++;
        }

        Matrix row_format(num_rows);

        for (size_t col_idx = col_start; col_idx < col_end; ++col_idx) {
            for (auto row_idx : col_format[col_idx]) {
                row_format[row_idx].push_back(col_idx);
            }
        }

        return row_format;
    }

    static Matrix col_to_row_format_parallel(const Matrix& col_format, int
        n_threads, size_t col_start = 0,
        size_t col_end = std::numeric_limits<size_t>::max(), Int num_rows = -1)
    {
        if (col_format.empty()) {
            return {};
        }

        if (col_end > col_format.size()) {
            col_end = col_format.size();
        }

        if (num_rows == -1) {
            for (const auto& col : col_format) {
                if (!col.empty()) {
                    num_rows = std::max(num_rows, col.back());
                }
            }
            num_rows++;
        }

        if (num_rows <= 0) {
            return {};
        }

        Matrix row_format(static_cast<size_t>(num_rows));

        if (col_start >= col_end) {
            return row_format;
        }

        const size_t n_cols = col_end - col_start;
        const size_t requested_threads = n_threads > 0 ? static_cast<size_t>(n_threads) : 1;
        const size_t n_workers = std::min(requested_threads, n_cols);

        if (n_workers == 0) {
            return row_format;
        }

        std::vector<std::vector<size_t>> per_thread_positions(
                n_workers, std::vector<size_t>(static_cast<size_t>(num_rows), 0));

        auto worker_range = [n_cols, n_workers, col_start](size_t tid) {
            const size_t begin = col_start + (tid * n_cols) / n_workers;
            const size_t end = col_start + ((tid + 1) * n_cols) / n_workers;
            return std::pair<size_t, size_t>(begin, end);
        };

        std::vector<std::thread> workers;
        workers.reserve(n_workers);

        for (size_t tid = 0; tid < n_workers; ++tid) {
            workers.emplace_back([&, tid]() {
                auto [begin, end] = worker_range(tid);
                auto& local_counts = per_thread_positions[tid];
                for (size_t col_idx = begin; col_idx < end; ++col_idx) {
                    for (int row_idx : col_format[col_idx]) {
                        ++local_counts[static_cast<size_t>(row_idx)];
                    }
                }
            });
        }

        for (auto& t : workers) {
            t.join();
        }

        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            size_t prefix = 0;
            const size_t r = static_cast<size_t>(row_idx);
            for (size_t tid = 0; tid < n_workers; ++tid) {
                const size_t count = per_thread_positions[tid][r];
                per_thread_positions[tid][r] = prefix;
                prefix += count;
            }
            row_format[r].resize(prefix);
        }

        workers.clear();

        for (size_t tid = 0; tid < n_workers; ++tid) {
            workers.emplace_back([&, tid]() {
                auto [begin, end] = worker_range(tid);
                auto& local_pos = per_thread_positions[tid];
                for (size_t col_idx = begin; col_idx < end; ++col_idx) {
                    for (int row_idx : col_format[col_idx]) {
                        const size_t r = static_cast<size_t>(row_idx);
                        row_format[r][local_pos[r]++] = static_cast<int>(col_idx);
                    }
                }
            });
        }

        for (auto& t : workers) {
            t.join();
        }

        return row_format;
    }

};

template<typename Int_, int P>
struct RVColumn {
    using Int = Int_;
    using FieldElement = Int_;
    using Entry = std::pair<Int_, FieldElement>;
    // generic (Z_p, P != 2) path: entries are (row, coeff) pairs and this
    // path is not the at-rest reduction column, so plain std::vector is fine.
    using Column = std::vector<Entry>;

    Column r_column;
    Column v_column;

    RVColumn() = default;
    RVColumn(const RVColumn& other) = default;
    RVColumn(RVColumn&& other) noexcept = default;
    RVColumn(Column&& _r, Column&& _v)
            :r_column(_r), v_column(_v) { }

    RVColumn(const Column& _r, const Column& _v)
            :r_column(_r), v_column(_v) { }

    [[nodiscard]] bool is_zero() const
    {
        return r_column.empty();
    }

    Int low() const
    {
        return is_zero() ? -1 : r_column.back().first;
    }

    void swap(RVColumn& other) noexcept
    {
        r_column.swap(other.r_column);
        v_column.swap(other.v_column);
    }

    template<class I>
    friend std::ostream& operator<<(std::ostream& out, const RVColumn<I, P>& rv);
};

template<typename Int_>
struct RVColumn<Int_, 2> {
    using Int = Int_;
    using Entry = Int;
    using Column = SparseColumn<Entry>;

    Column r_column;
    Column v_column;

    RVColumn() = default;
    RVColumn(const RVColumn& other) = default;
    RVColumn(RVColumn&& other) noexcept = default;
    RVColumn(Column&& _r, Column&& _v)
            :r_column(std::move(_r)), v_column(std::move(_v)) { }

    RVColumn(const Column& _r, const Column& _v)
            :r_column(_r), v_column(_v) { }

    [[nodiscard]] bool is_zero() const
    {
        return r_column.empty();
    }

    Int low() const
    {
        return is_zero() ? -1 : r_column.back();
    }

    void swap(RVColumn& other) noexcept
    {
        r_column.swap(other.r_column);
        v_column.swap(other.v_column);
    }

#ifdef OINEUS_USE_JEMALLOC
    // Route the heap-allocated working-column OBJECT to jemalloc. `new RVColumn`
    // and `delete p` (incl. via MemoryReclaim) auto-dispatch to these.
    static void* operator new(std::size_t n)
    {
        void* p = je_malloc(n);
        if (p == nullptr)
            throw std::bad_alloc();
        return p;
    }
    static void operator delete(void* p) noexcept { je_free(p); }
#endif

    template<class I>
    friend std::ostream& operator<<(std::ostream& out, const RVColumn<I, 2>& rv);
};


template<class I, int P>
std::ostream& operator<<(std::ostream& out, const RVColumn<I, P>& rv)
{
    out << "R column: [";
    for(auto x: rv.r_column)
        out << "(" << x.first << ": " << x.second << "), ";


    out << "], V column: [";
    for(auto x: rv.v_column)
        out << "(" << x.first << ": " << x.second << "), ";

    out << "]";

    return out;
}

template<class I>
std::ostream& operator<<(std::ostream& out, const RVColumn<I, 2>& rv)
{
    out << "R column: [";
    for(auto x: rv.r_column)
        out << x << ", ";


    out << "], V column: [";
    for(auto x: rv.v_column)
        out << x << ", ";

    out << "]";

    return out;
}

template<class I, int P>
std::ostream& operator<<(std::ostream& out, const std::atomic<RVColumn<I, P>*>& rv)
{
    out << *rv;
    return out;
}

template<class I>
std::ostream& operator<<(std::ostream& out, const std::atomic<RVColumn<I, 2>*>& rv)
{
    out << *rv;
    return out;
}

template<class I, int P>
std::ostream& operator<<(std::ostream& out, const std::vector<std::atomic<RVColumn<I, P>*>>& rvs)
{
    out << "RV matrix:\n";
    for(const auto& rv: rvs)
        out << rv << "\n";
    out << "\n";
    return out;
}

template<typename Int_, int P>
struct SimpleRVMatrixTraits {
    using Int = Int_;
};

template<typename Int_>
struct SimpleRVMatrixTraits<Int_, 2> {
    using Int = Int_;

    using Column = RVColumn<Int, 2>;
    using CachedColumn = std::pair<std::set<Int>, std::set<Int>>;
    using PColumn = Column*;
    using APColumn = std::atomic<PColumn>;
    using AMatrix = std::vector<APColumn>;

    static bool is_zero(const Column* col)       { return col->is_zero(); }
    static bool is_zero(const CachedColumn& col) { return col.first.empty(); }

    static Int low(const Column* col)       { return col->low(); }
    static Int low(const Column& col)       { return col.low(); }

    static Int low(const CachedColumn& col) { return is_zero(col) ? -1 : *col.first.rbegin(); }

    static size_t r_column_size(const Column& col) { return col.r_column.size(); }
    static size_t v_column_size(const Column& col) { return col.v_column.size(); }

    static size_t r_column_size(const CachedColumn& col) { return col.first.size(); }
    static size_t v_column_size(const CachedColumn& col) { return col.second.size(); }

    static void add_to_cached(const Column& pivot, CachedColumn& reduced)
    {
        SimpleSparseMatrixTraits<Int, 2>::add_to_cached(pivot.r_column, reduced.first);
        SimpleSparseMatrixTraits<Int, 2>::add_to_cached(pivot.v_column, reduced.second);
    }

    static void add_to_cached(const Column* pivot, CachedColumn& reduced)
    {
        add_to_cached(*pivot, reduced);
    }

    static CachedColumn load_to_cache(const Column& col)
    {
        return {{col.r_column.begin(), col.r_column.end()}, {col.v_column.begin(), col.v_column.end()}};
    }

    static CachedColumn load_to_cache(Column* col)
    {
        if (col == nullptr)
            return {{}, {}};
        else
            return load_to_cache(*col);
    }

    static PColumn load_from_cache(const CachedColumn& col)
    {
        return new Column({col.first.begin(), col.first.end()}, {col.second.begin(), col.second.end()});
    }

    static auto& r_data(Column* col) { return col->r_column; }

    static std::string check_col_duplicates(PColumn col)
    {
        if (col == nullptr) return "";
        std::string s_r = SimpleSparseMatrixTraits<Int, 2>::check_col_duplicates(&(col->r_column));
        std::string s_v = SimpleSparseMatrixTraits<Int, 2>::check_col_duplicates(&(col->v_column));
        return s_r + s_v;
    }
//    // get identity matrix
//    static Matrix eye(size_t n)
//    {
//        Matrix cols{n};
//
//        for(Int i = 0 ; i < n ; ++i)
//            cols[i].emplace_back(i);
//        return cols;
//    }


};


template<typename Int_>
struct SparseMatrix {
    // types
    using Int = Int_;
    using Column = SparseColumn<Int>;
    using Data = std::vector<Column>;

    template<class I> friend class VRUDecomposition;

    //methods

    SparseMatrix(size_t n_rows, size_t n_cols)
            :n_rows_(n_rows), n_cols_(n_cols), row_data_(n_rows), col_data_(n_cols) { }

    SparseMatrix(const Data& data, size_t n_other, bool is_col)
            :
            n_rows_(is_col ? n_other : data.size()),
            n_cols_(is_col ? data.size() : n_other),
            row_data_(is_col ? Data() : data),
            col_data_(is_col ? data : Data()) { if (is_col) compute_rows(); else compute_cols(); }

    SparseMatrix(Data&& col_data, size_t n_rows)
            :n_rows_(n_rows), n_cols_(col_data.size()), col_data_(col_data) { compute_rows(); }

    SparseMatrix(SparseMatrix<Int_>& R) = default;

    bool is_row_zero(size_t row_idx) const { return row_data_[row_idx].empty(); }
    bool is_col_zero(size_t col_idx) const { return col_data_[col_idx].empty(); }

    Column& row(size_t row_idx) { return row_data_[row_idx]; }
    const Column& row(size_t row_idx) const { return row_data_[row_idx]; }
    Column& col(size_t col_idx) { return col_data_[col_idx]; }
    const Column& col(size_t col_idx) const { return col_data_[col_idx]; }

    size_t n_rows() const { return n_rows_; }
    size_t n_cols() const { return n_cols_; }

    // in place
    void transpose()
    {
        std::swap(row_data_, col_data_);
        std::swap(n_rows_, n_cols_);
    }

    // compute row data
    void compute_rows()
    {
        row_data_ = Data(n_rows_);
        for(Int col_idx = 0 ; col_idx < static_cast<Int>(col_data_.size()) ; ++col_idx) {
            for(auto&& e: col_data_[col_idx]) {
                row_data_[e].push_back(col_idx);
            }
        }

        assert(sanity_check());
    }

    // compute columns data
    void compute_cols()
    {
        col_data_ = Data(n_cols());
        for(Int row_idx = 0 ; row_idx < static_cast<Int>(row_data_.size()) ; ++row_idx) {
            for(auto&& e: row_data_[row_idx]) {
                col_data_[e].push_back(row_idx);
            }
        }

        assert(sanity_check());
    }

    bool operator==(const SparseMatrix& other) const
    {
        assert(sanity_check() and other.sanity_check());
        return col_data_ == other.col_data_;
    }

    bool operator!=(const SparseMatrix& other) const
    {
        return !(col_data_ == other.col_data_);
    }

    bool is_lower_triangular() const
    {
        for(size_t col_idx = 0 ; col_idx < n_cols() ; ++col_idx) {
            auto& col = col_data_[col_idx];
            if (not col.empty() and col.front() < col_idx)
                return false;
        }
        return true;
    }

    bool sanity_check() const
    {
        if (not std::all_of(row_data_.begin(), row_data_.end(), [](const Column& v) { return std::is_sorted(v.begin(), v.end()); }))
            return false;

        if (not std::all_of(col_data_.begin(), col_data_.end(), [](const Column& v) { return std::is_sorted(v.begin(), v.end()); }))
            return false;

        if (not std::all_of(col_data_.begin(), col_data_.end(),
                [this](const Column& v) {
                  return std::all_of(v.begin(), v.end(), [this](Int e) { return e >= 0 and e < static_cast<Int>(this->n_rows()); });
                }))
            return false;

        if (not std::all_of(row_data_.begin(), row_data_.end(),
                [this](const Column& v) {
                  return std::all_of(v.begin(), v.end(), [this](Int e) { return e >= 0 and e < static_cast<Int>(this->n_cols()); });
                }))
            return false;

        std::set<std::pair<size_t, size_t>> e_1, e_2;

        for(size_t col_idx = 0 ; col_idx < n_cols() ; ++col_idx)
            for(Int row_idx: col_data_[col_idx])
                e_1.emplace(row_idx, col_idx);

        for(size_t row_idx = 0 ; row_idx < n_rows() ; ++row_idx)
            for(Int col_idx: row_data_[row_idx])
                e_2.emplace(row_idx, col_idx);

        return e_1 == e_2;
    }

    Column get_col(int col_id)
    {
        return col_data_[col_id];
    }

    void update_col(int col_id, Column new_col)
    {
        col_data_[col_id] = new_col;
    }

    void delete_col(int col_ind)
    {
        col_data_.erase(col_ind);
    }

    void delete_cols(std::vector<int> cols_to_del)
    {
        for(size_t i = 0 ; i < cols_to_del.size() ; i++) {
            delete_col(cols_to_del[i]);
        }

        assert(sanity_check());
    }

    // compute columns data
    void reorder_rows(std::vector<int> new_order)
    {
        for(int i = 0 ; i < n_cols() ; i++) {
            Column new_col;
            for(int j = 0 ; j < get_col(i).size() ; j++) {
                new_col.push_back(new_order[get_col(i)[j]]);
            }
            update_col(i, new_col);
        }

        assert(sanity_check());
    }

    template<class I>
    friend SparseMatrix<I> mat_multiply_2(const SparseMatrix<I>& a, const SparseMatrix<I>& b);

    template<class I>
    friend SparseMatrix<I> antitranspose(const SparseMatrix<I>& a);

    bool is_upper_triangular() const
    {
        for(size_t col_idx = 0 ; col_idx < n_cols() ; ++col_idx) {
            auto& col = col_data_[col_idx];
            if (not col.empty() and col.back() > static_cast<Int>(col_idx))
                return false;
        }
        return true;
    }
private:
    size_t n_rows_ {0};
    size_t n_cols_ {0};

    Data row_data_;
    Data col_data_;
};

template<typename Int>
SparseMatrix<Int> eye(size_t n)
{
    typename SparseMatrix<Int>::Data cols(n);

    for(size_t i = 0 ; i < n ; ++i)
        cols[i].push_back(i);

    return SparseMatrix<Int>(cols, n, true);
}

template<typename Int>
SparseMatrix<Int> mat_multiply_2(const SparseMatrix<Int>& a, const SparseMatrix<Int>& b)
{
    if (a.n_cols() != b.n_rows())
        throw std::runtime_error("dimension mismatch, n_cols != n_rows");

    SparseMatrix<Int> c(a.n_rows(), b.n_cols());

    for(size_t row_idx = 0 ; row_idx < a.n_rows() ; ++row_idx) {
        if (a.is_row_zero(row_idx))
            continue;

        for(size_t col_idx = 0 ; col_idx < b.n_cols() ; ++col_idx) {
            if (b.is_col_zero(col_idx))
                continue;

            // compute entry c[row_idx][col_idx]

            Int c_ij = 0;
            auto a_iter = a.row(row_idx).cbegin();
            auto b_iter = b.col(col_idx).cbegin();

            while(a_iter != a.row(row_idx).cend() and b_iter != b.col(col_idx).cend()) {
                if (*a_iter < *b_iter)
                    ++a_iter;
                else if (*b_iter < *a_iter)
                    ++b_iter;
                else {
                    assert(*a_iter == *b_iter);
                    // for char != 2: replace with c_ij += a_ik * b_kj
                    c_ij += 1;
                    ++a_iter;
                    ++b_iter;
                }
            }

            if (c_ij % 2) {
                c.row_data_[row_idx].push_back(col_idx);
                c.col_data_[col_idx].push_back(row_idx);
            }
        }
    }

    return c;
}

template<typename Int>
SparseMatrix<Int> antitranspose(const SparseMatrix<Int>& a)
{
    CALI_CXX_MARK_FUNCTION;
    SparseMatrix<Int> result(a.n_cols(), a.n_rows());

    for(size_t row_idx = 0 ; row_idx < a.size() ; ++row_idx) {
        const auto& row = a.row(a.n_rows() - 1 - row_idx);
        for(auto c = row.rbegin() ; c != row.rend() ; ++c) {
            size_t col_idx = a.n_cols() - 1 - *c;
            result.col_data_[col_idx].push_back(row_idx);
            result.row_data_[row_idx].push_back(col_idx);
        }
    }

    assert(result.sanity_check());

    return result;
}

template<typename Int>
std::ostream& operator<<(std::ostream& out, const typename SparseMatrix<Int>::Column& c)
{
    out << "[";
    for(int i = 0 ; i < c.size() ; ++i) {
        out << c[i];
        if (i != c.size() - 1)
            out << ", ";
    }
    out << "]";
    return out;
}

template<typename Int>
std::ostream& operator<<(std::ostream& out, const SparseMatrix<Int>& m)
{
    out << "Matrix(n_rows = " << m.n_rows() << ", n_cols = " << m.n_cols() << "\n[";
    for(size_t col_idx = 0 ; col_idx < m.n_cols() ; ++col_idx) {
        const auto& c = m.col(col_idx);
        out << "[";
        for(size_t i = 0 ; i < c.size() ; ++i) {
            out << c[i];
            if (i + 1 != c.size())
                out << ", ";
        }
        out << "]\n";
    }
    out << "]\n";
    return out;
}

// Container-generic: works for both std::vector<std::vector<Int>> and the
// at-rest std::vector<SparseColumn<Int>> (MatrixData). Returns the same
// matrix/column type as the input so the result assigns straight into d_data.
//
// The serial path is a single scatter that emits each column's entries in
// increasing row order (so result columns come out sorted for free). For
// cohomology of large packed (VR/alpha) complexes this scatter is the serial
// bottleneck of coboundary construction (the boundary build around it is
// parallel), so n_threads>1 runs a lock-free counting-sort transpose instead:
// count per-result-column degrees, allocate exactly, scatter into preallocated
// slots via a per-column atomic cursor (distinct slots -> no data race), then
// sort each (small) column. Same result as the serial path.
template<typename Matrix>
Matrix antitranspose(const Matrix& a, size_t n_rows, int n_threads = 1)
{
    using Int = typename Matrix::value_type::value_type;
    const size_t n_cols = a.size();
    Matrix result(n_cols);

    if (n_threads <= 1 or n_cols < static_cast<size_t>(64) * static_cast<size_t>(n_threads)) {
        for(size_t row_idx = 0 ; row_idx < n_cols ; ++row_idx) {
            for(Int c: a[n_rows - 1 - row_idx]) {
                size_t col_idx = n_cols - 1 - static_cast<size_t>(c);
                result[col_idx].push_back(static_cast<Int>(row_idx));
            }
        }
        return result;
    }

    // deg[j]: first the entry count of result column j, then reused as the
    // atomic write cursor during scatter.
    std::unique_ptr<std::atomic<size_t>[]> deg(new std::atomic<size_t>[n_cols]);

    tf::Executor executor(n_threads);

    tf::Taskflow tf_count;
    tf_count.for_each_index((size_t)0, n_cols, (size_t)1, [&](size_t j) {
        deg[j].store(0, std::memory_order_relaxed);
    });
    executor.run(tf_count).wait();

    tf::Taskflow tf_deg;
    tf_deg.for_each_index((size_t)0, n_cols, (size_t)1, [&](size_t row_idx) {
        for(Int c: a[n_rows - 1 - row_idx])
            deg[n_cols - 1 - static_cast<size_t>(c)].fetch_add(1, std::memory_order_relaxed);
    });
    executor.run(tf_deg).wait();

    tf::Taskflow tf_alloc;
    tf_alloc.for_each_index((size_t)0, n_cols, (size_t)1, [&](size_t j) {
        result[j].resize(deg[j].load(std::memory_order_relaxed));
        deg[j].store(0, std::memory_order_relaxed);   // reuse as write cursor
    });
    executor.run(tf_alloc).wait();

    tf::Taskflow tf_scatter;
    tf_scatter.for_each_index((size_t)0, n_cols, (size_t)1, [&](size_t row_idx) {
        const Int row_val = static_cast<Int>(row_idx);
        for(Int c: a[n_rows - 1 - row_idx]) {
            size_t j = n_cols - 1 - static_cast<size_t>(c);
            size_t pos = deg[j].fetch_add(1, std::memory_order_relaxed);
            result[j][pos] = row_val;
        }
    });
    executor.run(tf_scatter).wait();

    tf::Taskflow tf_sort;
    tf_sort.for_each_index((size_t)0, n_cols, (size_t)1, [&](size_t j) {
        std::sort(result[j].begin(), result[j].end());
    });
    executor.run(tf_sort).wait();

    return result;
}

template<typename Matrix>
Matrix transpose(const Matrix& col_format, int num_rows = -1, int n_threads = 1)
{
    using Column = typename Matrix::value_type;
    using Int = typename Column::value_type;
    if (col_format.empty()) {
        return {};
    }

    // Determine the number of rows needed, if not given as parameter
    if (num_rows < 0) {
        num_rows = 0;
        for (const auto& col : col_format) {
            if (!col.empty()) {
                num_rows = std::max(num_rows, col.back() + 1);
            }
        }
    }

    Matrix row_format(num_rows);

    if (n_threads <= 1 or col_format.size() < 20 * n_threads) {
        // Iterate through each column
        for (size_t col_idx = 0; col_idx < col_format.size(); ++col_idx) {
            // For each non-zero entry in this column
            for (Int row_idx : col_format[col_idx]) {
                row_format[static_cast<size_t>(row_idx)].push_back(static_cast<Int>(col_idx));
            }
        }
    } else {
        const int num_cols = static_cast<int>(col_format.size());

        // Create a temporary structure: for each thread, store data for each row
        // This avoids race conditions during parallel insertion
        std::vector<std::vector<std::vector<Int>>> temp_storage(n_threads);
        for (auto& storage : temp_storage) {
            storage.resize(num_rows);
        }

        // Create executor with specified number of threads
        tf::Executor executor(n_threads);
        tf::Taskflow taskflow;

        // Parallel phase: each thread processes a subset of columns
        taskflow.for_each_index(0, num_cols, 1,
            [&](int col_idx) {
                int worker_id = executor.this_worker_id();
                if (worker_id < 0) worker_id = 0; // fallback

                // Add column index to appropriate rows in this thread's local storage
                for (Int row_idx : col_format[col_idx]) {
                    temp_storage[worker_id][static_cast<size_t>(row_idx)].push_back(static_cast<Int>(col_idx));
                }
            }
        );

        executor.run(taskflow).wait();

        // Merge phase: combine results from all threads
        tf::Taskflow merge_taskflow;
        merge_taskflow.for_each_index(0, num_rows, 1,
            [&](int row_idx) {
                // Calculate total size for this row
                size_t total_size = 0;
                for (const auto& storage : temp_storage) {
                    total_size += storage[static_cast<size_t>(row_idx)].size();
                }

                // Reserve space and merge
                row_format[static_cast<size_t>(row_idx)].reserve(total_size);
                for (const auto& storage : temp_storage) {
                    row_format[static_cast<size_t>(row_idx)].insert(
                        row_format[static_cast<size_t>(row_idx)].end(),
                        storage[static_cast<size_t>(row_idx)].begin(),
                        storage[static_cast<size_t>(row_idx)].end()
                    );
                }

                // Sort to maintain sorted order
                std::sort(row_format[static_cast<size_t>(row_idx)].begin(), row_format[static_cast<size_t>(row_idx)].end());
            }
        );
        executor.run(merge_taskflow).wait();
    }

    return row_format;
}

template<typename Int>
std::vector<std::vector<std::pair<Int, Int>>> transpose_and_densify_for_targets(const std::vector<std::vector<Int>>& r_col_data, const std::unordered_set<Int>& row_indices, int num_rows = -1)
{
    int n_threads = 1;
    if (r_col_data.empty()) {
        return {};
    }

    // Determine the number of rows needed, if not given as parameter
    if (num_rows < 0) {
        num_rows = 0;
        for (const auto& col : r_col_data) {
            if (!col.empty()) {
                num_rows = std::max(num_rows, static_cast<decltype(num_rows)>(col.back() + 1));
            }
        }
    }

    std::vector<std::vector<std::pair<Int, Int>>> row_format(num_rows);

    // pre-allocate memory for rows
    double _avg_size = 0;
    for(auto& col : r_col_data) {
        _avg_size += double(col.size()) / (double)r_col_data.size();
    }

    size_t avg_size = 3 * ceil(_avg_size) / 2;

    for(Int row_idx = 0; row_idx < static_cast<Int>(num_rows); ++row_idx) {
        if (row_indices.find(row_idx) == row_indices.end()) {
            row_format[row_idx].reserve(avg_size);
        } else {
            row_format[row_idx].reserve(r_col_data.size() / 2);
        }
    }

    if (n_threads <= 1 or r_col_data.size() < 20 * n_threads) {
        // Iterate through each column
        for (int col_idx = 0; col_idx < r_col_data.size(); ++col_idx) {
            // For each non-zero entry in this column
            if (row_indices.find(col_idx) == row_indices.end()) {
                // we don't want that row of U back from Pardiso, just normal transformation to row format
                for (int row_idx : r_col_data[col_idx]) {
                    row_format[row_idx].emplace_back(col_idx, 1);
                }
            } else {
                std::unordered_set<Int> row_indices_in_col(r_col_data[col_idx].begin(), r_col_data[col_idx].end());
                for(Int row_idx = col_idx; row_idx < r_col_data.size(); row_idx++) {
                        Int entry = row_indices_in_col.find(row_idx) != row_indices_in_col.end();
                        row_format[row_idx].emplace_back(col_idx, entry);
                }
            }
        }
    } else {
        throw std::runtime_error("not implemented yet");
    }
    return row_format;
}

} // namespace oineus

#endif //OINEUS_SPARSE_MATRIX_H
