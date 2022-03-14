#pragma once
#ifndef OINEUS_SPARSE_MATRIX_H
#define OINEUS_SPARSE_MATRIX_H

#include <vector>
#include <utility>
#include <set>
#include <unordered_set>
#include <cassert>
#include <algorithm>

#include "icecream/icecream.hpp"


namespace oineus {

// contains indices of non-zero entries
template<typename Int>
using SparseColumn = std::vector<Int>;

//template<typename Int>
//class VRUDecomposition;

template<typename Int_>
struct SparseMatrix {
    // types
    using Int = Int_;
    using Column = SparseColumn<Int>;
    using Data = std::vector<Column>;

    template<class I> friend class VRUDecomposition;

    //methods

    SparseMatrix(size_t n_rows, size_t n_cols) : n_rows_(n_rows), n_cols_(n_cols), col_data_(n_cols), row_data_(n_rows) {}
    SparseMatrix(const Data& data, size_t n_other, bool is_col) :
        n_rows_(is_col ? n_other : data.size()),
        n_cols_(is_col ? data.size() : n_other),
        col_data_(is_col ? data : Data()),
        row_data_(is_col ? Data() : data)
    { if (is_col) compute_rows(); else compute_cols(); }

    SparseMatrix(Data&& col_data, size_t n_rows) : n_rows_(n_rows), n_cols_(col_data.size()), col_data_(col_data) { compute_rows(); }

    bool is_row_zero(size_t row_idx) const { return row_data_[row_idx].empty(); }
    bool is_col_zero(size_t col_idx) const { return col_data_[col_idx].empty(); }

    Column&       row(size_t row_idx)       { return row_data_[row_idx]; }
    const Column& row(size_t row_idx) const { return row_data_[row_idx]; }
    Column&       col(size_t col_idx)       { return col_data_[col_idx]; }
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
        for(Int col_idx = 0; col_idx < static_cast<Int>(col_data_.size()); ++col_idx) {
            for(auto&& e : col_data_[col_idx]) {
                row_data_[e].push_back(col_idx);
            }
        }

        assert(sanity_check());
    }

    // compute columns data
    void compute_cols()
    {
        col_data_ = Data(n_cols());
        for(Int row_idx = 0; row_idx < static_cast<Int>(row_data_.size()); ++row_idx) {
            for(auto&& e : row_data_[row_idx]) {
                col_data_[e].push_back(row_idx);
            }
        }

        assert(sanity_check());
    }

    template<class I>
    friend SparseMatrix<I> mat_multiply_2(const SparseMatrix<I>& a, const SparseMatrix<I>& b);

    template<class I>
    friend SparseMatrix<I> antitranspose(const SparseMatrix<I>& a);

    bool is_upper_triangular() const
    {
        for(size_t col_idx = 0; col_idx < n_cols(); ++col_idx) {
            auto& col = col_data_[col_idx];
            if (not col.empty() and col.back() > col_idx)
                return false;
        }
        return true;
    }

    bool operator==(const SparseMatrix& other) const
    {
        assert(sanity_check() and other.sanity_check());
        return col_data_ == other.col_data_;
    }

    bool is_lower_triangular() const
    {
        for(size_t col_idx = 0; col_idx < n_cols(); ++col_idx) {
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
                [this](const Column& v)
                {
                    return std::all_of(v.begin(), v.end(), [this](Int e) { return e >= 0 and e < this->n_rows(); });
                }))
            return false;

        if (not std::all_of(row_data_.begin(), row_data_.end(),
                [this](const Column& v)
                {
                    return std::all_of(v.begin(), v.end(), [this](Int e) { return e >= 0 and e < this->n_cols(); });
                }))
            return false;

        std::set<std::pair<size_t, size_t>> e_1, e_2;

        for(size_t col_idx = 0; col_idx < n_cols(); ++col_idx)
            for(Int row_idx : col_data_[col_idx])
                e_1.emplace(row_idx, col_idx);

        for(size_t row_idx = 0; row_idx < n_rows(); ++row_idx)
            for(Int col_idx : row_data_[row_idx])
                e_2.emplace(row_idx, col_idx);

        return e_1 == e_2;
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

    for(size_t i = 0; i < n; ++i)
        cols[i].push_back(i);

    return SparseMatrix<Int>(cols, n, true);
}

template<typename Int>
SparseMatrix<Int> mat_multiply_2(const SparseMatrix<Int>& a, const SparseMatrix<Int>& b)
{
    if (a.n_cols() != b.n_rows())
        throw std::runtime_error("dimension mismatch, n_cols != n_rows");

    SparseMatrix<Int> c(a.n_rows(), b.n_cols());

    for(size_t row_idx = 0; row_idx < a.n_rows(); ++row_idx) {
        if (a.is_row_zero(row_idx))
            continue;

        for(size_t col_idx = 0; col_idx < b.n_cols(); ++col_idx) {
//            IC(col_idx, row_idx);
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
//                IC(col_idx, row_idx, c_ij);
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
    SparseMatrix<Int> result(a.n_cols(), a.n_rows());

    for(size_t row_idx = 0; row_idx < a.size(); ++row_idx) {
        const auto& row = a.row(a.n_rows() - 1 - row_idx);
        for(auto c = row.rbegin(); c != row.rend(); ++c) {
            size_t col_idx = a.n_cols() - 1 - *c;
            result.col_data_[col_idx].push_back(row_idx);
            result.row_data_[row_idx].push_back(col_idx);
        }
    }

    assert(result.sanity_check());

    return result;
}


template<typename Int>
std::vector<SparseColumn<Int>> antitranspose(const std::vector<SparseColumn<Int>>& a, size_t n_rows)
{
    std::vector<SparseColumn<Int>> result {a.size(), SparseColumn<Int>() };
    for(size_t row_idx = 0; row_idx < a.size(); ++row_idx) {
        for(auto c : a[n_rows - 1 - row_idx]) {
            size_t col_idx = a.size() - 1 - c;
            result[col_idx].push_back(row_idx);
        }
    }
    return result;
}

} // namespace oineus

#endif //OINEUS_SPARSE_MATRIX_H
