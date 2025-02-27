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

#include "profile.h"

namespace oineus {

template<class Int>
using SparseColumn = std::vector<Int>;

template<class Int>
bool is_zero(const SparseColumn<Int>& col)
{
    return col.empty();
}

template<class Int>
bool is_zero(const SparseColumn<Int>* col)
{
    return col->empty();
}

template<class Int>
Int low(const SparseColumn<Int>* col)
{
    return is_zero(col) ? -1 : col->back();
}

template<class Int>
Int low(const SparseColumn<Int>& col)
{
    return is_zero(col) ? -1 : col.back();
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

    using Column = std::vector<Entry>;
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

    static bool is_zero(const CachedColumn& col)
    {
        return col.empty();
    }

    static Int low(const CachedColumn& col)
    {
        return is_zero(col) ? -1 : *col.rbegin();
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
};

template<typename Int_, int P>
struct RVColumn {
    using Int = Int_;
    using FieldElement = Int_;
    using Entry = std::pair<Int_, FieldElement>;
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
        return is_zero() ? -1 : r_column.back();
    }

    void swap(RVColumn& other) noexcept
    {
        r_column.swap(other.r_column);
        v_column.swap(other.v_column);
    }

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
    using Column = std::vector<Int>;
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
            std::vector<int> new_col;
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
        out << m.col(col_idx) << "\n";
    }
    out << "]\n";
    return out;
}

template<typename Int>
std::vector<std::vector<Int>> antitranspose(const std::vector<std::vector<Int>>& a, size_t n_rows)
{
    using SparseColumn = std::vector<Int>;
    std::vector<SparseColumn> result {a.size(), SparseColumn()};
    for(size_t row_idx = 0 ; row_idx < a.size() ; ++row_idx) {
        for(auto c: a[n_rows - 1 - row_idx]) {
            size_t col_idx = a.size() - 1 - c;
            result[col_idx].push_back(row_idx);
        }
    }
    return result;
}

} // namespace oineus

#endif //OINEUS_SPARSE_MATRIX_H
