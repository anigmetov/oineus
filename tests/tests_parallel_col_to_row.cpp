// Tests for the PRODUCTION parallel col->row transpose,
// oineus::SimpleSparseMatrixTraits<Int, 2>::col_to_row_format_parallel
// (include/oineus/sparse_matrix.h): parallel count + parallel per-row prefix
// sum + scatter, where the scatter strategy is picked by the caller via
// prefer_row_scatter (column-partitioned vs row-partitioned), plus the
// row-bucket radix variant col_to_row_format_bucket. Every case is checked
// against a simple serial reference transpose defined here, for all three
// strategies and several thread counts.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <oineus/sparse_matrix.h>

namespace {

template<class Int>
using Matrix = typename oineus::SimpleSparseMatrixTraits<Int, 2>::Matrix;

// Simple serial reference transpose with the same interface/defaults as the
// production function: clamp col_end, derive num_rows from ALL columns when
// not given, entries within each row list ascend (columns are walked in
// increasing order and each column is sorted).
template<class Int>
Matrix<Int> col_to_row_reference(const Matrix<Int>& col_format, size_t col_start = 0,
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
                num_rows = std::max(num_rows, static_cast<Int>(col.back()));
            }
        }
        num_rows++;
    }

    if (num_rows <= 0) {
        return {};
    }

    Matrix<Int> row_format(static_cast<size_t>(num_rows));

    for (size_t col_idx = std::min(col_start, col_end); col_idx < col_end; ++col_idx) {
        for (Int row_idx : col_format[col_idx]) {
            row_format[static_cast<size_t>(row_idx)].push_back(static_cast<Int>(col_idx));
        }
    }

    return row_format;
}

// generic on the matrix type: Matrix<Int> is an alias template, so Int is
// not deducible through it
template<class M>
bool rows_are_sorted(const M& rows)
{
    for (const auto& row : rows) {
        if (!std::is_sorted(row.begin(), row.end())) {
            return false;
        }
    }
    return true;
}

// Run the production transpose across thread counts and ALL THREE scatter
// strategies and require the result to match the serial reference.
template<class Int>
void check_against_reference(const Matrix<Int>& cols, size_t col_start = 0,
        size_t col_end = std::numeric_limits<size_t>::max(), Int num_rows = -1)
{
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;

    const Matrix<Int> expected = col_to_row_reference<Int>(cols, col_start, col_end, num_rows);
    REQUIRE(rows_are_sorted(expected));

    for (int threads : {1, 2, 4, 8}) {
        for (bool prefer_row_scatter : {false, true}) {
            const Matrix<Int> got = MatrixTraits::col_to_row_format_parallel(
                    cols, threads, col_start, col_end, num_rows, prefer_row_scatter);
            REQUIRE(got == expected);
        }
        const Matrix<Int> got_bucket = MatrixTraits::col_to_row_format_bucket(
                cols, threads, col_start, col_end, num_rows);
        REQUIRE(got_bucket == expected);
    }
}

template<class Int>
Matrix<Int> make_sparse_columns(size_t n_cols, Int n_rows, int max_nnz_per_col,
        double nonempty_col_prob, uint64_t seed)
{
    Matrix<Int> cols(n_cols);

    if (n_rows <= 0 || max_nnz_per_col <= 0) {
        return cols;
    }

    std::mt19937_64 rng(seed);
    std::bernoulli_distribution is_nonempty(nonempty_col_prob);
    std::geometric_distribution<int> extra_nnz(0.7);
    std::uniform_int_distribution<Int> row_dist(0, n_rows - 1);

    std::vector<Int> tmp;
    tmp.reserve(static_cast<size_t>(max_nnz_per_col));

    for (auto& col : cols) {
        int target_nnz = 0;
        if (is_nonempty(rng)) {
            const int capped_extra = std::min(max_nnz_per_col - 1, extra_nnz(rng));
            target_nnz = 1 + capped_extra;
        }
        tmp.clear();
        while (static_cast<int>(tmp.size()) < target_nnz) {
            const Int row_idx = row_dist(rng);
            if (std::find(tmp.begin(), tmp.end(), row_idx) == tmp.end()) {
                tmp.push_back(row_idx);
            }
        }
        std::sort(tmp.begin(), tmp.end());
        col.assign(tmp.begin(), tmp.end());
    }

    return cols;
}

} // namespace

TEST_CASE("Parallel col->row: empty and trivial inputs")
{
    using Int = int;
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;

    // no columns at all
    {
        Matrix<Int> cols;
        for (bool prs : {false, true}) {
            REQUIRE(MatrixTraits::col_to_row_format_parallel(
                    cols, 4, 0, std::numeric_limits<size_t>::max(), Int(-1), prs).empty());
        }
        REQUIRE(MatrixTraits::col_to_row_format_bucket(
                cols, 4, 0, std::numeric_limits<size_t>::max(), Int(-1)).empty());
    }

    // columns present but zero rows (all columns empty), num_rows derived and explicit
    {
        Matrix<Int> cols(10);
        for (bool prs : {false, true}) {
            REQUIRE(MatrixTraits::col_to_row_format_parallel(
                    cols, 8, 0, std::numeric_limits<size_t>::max(), Int(-1), prs).empty());
        }
        REQUIRE(MatrixTraits::col_to_row_format_bucket(
                cols, 8, 0, std::numeric_limits<size_t>::max(), Int(-1)).empty());
        check_against_reference<Int>(cols, 0, std::numeric_limits<size_t>::max(), Int(7));
    }

    // a single column
    {
        Matrix<Int> cols(1);
        cols[0].assign({0, 2, 5});
        check_against_reference<Int>(cols);
    }

    // small hand-made matrix with an empty column in the middle
    {
        Matrix<Int> cols(4);
        cols[0].assign({0, 2, 4});
        cols[1].assign({1, 2});
        cols[3].assign({0, 3, 4});
        check_against_reference<Int>(cols);
    }
}

TEST_CASE("Parallel col->row: fewer columns than threads")
{
    using Int = int;

    Matrix<Int> cols(3);
    cols[0].assign({1, 5});
    cols[1].assign({0});
    cols[2].assign({1, 3, 5});
    // check_against_reference runs up to 8 threads on these 3 columns
    check_against_reference<Int>(cols);
}

TEST_CASE("Parallel col->row: randomized correctness across shapes and strategies")
{
    using Int = int;

    std::mt19937_64 rng(1234567);
    std::uniform_int_distribution<int> cols_dist(0, 300);
    std::uniform_int_distribution<Int> rows_dist(0, 400);
    std::uniform_int_distribution<int> nnz_dist(2, 4);
    std::uniform_real_distribution<double> nonempty_prob_dist(0.03, 0.20);

    for (int trial = 0; trial < 40; ++trial) {
        const size_t n_cols = static_cast<size_t>(cols_dist(rng));
        const Int n_rows = rows_dist(rng);
        const int max_nnz = nnz_dist(rng);
        const double nonempty_prob = nonempty_prob_dist(rng);

        Matrix<Int> cols = make_sparse_columns<Int>(
                n_cols, n_rows, max_nnz, nonempty_prob, 1000u + static_cast<uint64_t>(trial));
        check_against_reference<Int>(cols);
    }

    // degenerate shapes on purpose: 0 columns, 1 column, 1 row
    check_against_reference<Int>(make_sparse_columns<Int>(0, 100, 3, 0.5, 1));
    check_against_reference<Int>(make_sparse_columns<Int>(1, 100, 3, 1.0, 2));
    check_against_reference<Int>(make_sparse_columns<Int>(50, 1, 1, 0.5, 3));
}

TEST_CASE("Parallel col->row: dense rows")
{
    using Int = int;

    // a few rows appear in (almost) every column, so single output rows get
    // contributions from every worker's column block
    const size_t n_cols = 500;
    const Int n_rows = 64;
    Matrix<Int> cols = make_sparse_columns<Int>(n_cols, n_rows, 3, 0.3, 77);
    for (size_t c = 0; c < n_cols; ++c) {
        std::vector<Int> merged(cols[c].begin(), cols[c].end());
        for (Int dense_row : {Int(0), Int(31), Int(63)}) {
            if (std::find(merged.begin(), merged.end(), dense_row) == merged.end()) {
                merged.push_back(dense_row);
            }
        }
        std::sort(merged.begin(), merged.end());
        cols[c].assign(merged.begin(), merged.end());
    }
    check_against_reference<Int>(cols);
}

TEST_CASE("Parallel col->row: long Int instantiation")
{
    // the Python bindings instantiate the traits with long; make sure that
    // instantiation is exercised, not just int
    using Int = long;

    Matrix<Int> cols = make_sparse_columns<Int>(200, Int(250), 4, 0.15, 20260703);
    check_against_reference<Int>(cols);
}

TEST_CASE("Parallel col->row: subrange semantics match serial reference")
{
    using Int = int;

    Matrix<Int> cols(6);
    cols[1].assign({0, 2, 5});
    cols[2].assign({1});
    cols[3].assign({0, 4});
    cols[5].assign({3, 5});

    // proper subrange with explicit num_rows
    check_against_reference<Int>(cols, 1, 5, Int(6));
    // col_end past the end is clamped
    check_against_reference<Int>(cols, 2, 1000, Int(6));
    // empty subrange (col_start == col_end) yields all-empty rows
    check_against_reference<Int>(cols, 3, 3, Int(6));
}

TEST_CASE("Parallel col->row: medium sparse stress")
{
    using Int = int;

    const size_t n_cols = 10000;
    const Int n_rows = 12000;

    Matrix<Int> cols = make_sparse_columns<Int>(n_cols, n_rows, 3, 0.08, 42);
    check_against_reference<Int>(cols);
}

TEST_CASE("Parallel col->row: boundary-like 200k matrix per-dimension block conversion")
{
    using Int = int;
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;

    const size_t n_cols = 200000;
    const Int num_rows = static_cast<Int>(n_cols);
    const int max_nnz = 3;

    // Exclusive ranges for dimensions 0..3.
    const std::vector<size_t> dim_first{0, 50000, 110000, 160000};
    const std::vector<size_t> dim_last{50000, 110000, 160000, 200000};

    Matrix<Int> cols(n_cols);
    std::mt19937_64 rng(20260219);
    std::bernoulli_distribution is_nonempty(0.05);
    std::geometric_distribution<int> extra_nnz(0.7);

    // dim d>0: nonzero rows only in previous dimension range; dim 0 stays empty.
    for (size_t dim = 1; dim < dim_first.size(); ++dim) {
        const Int row_lo = static_cast<Int>(dim_first[dim - 1]);
        const Int row_hi = static_cast<Int>(dim_last[dim - 1] - 1);
        std::uniform_int_distribution<Int> row_dist(row_lo, row_hi);

        std::vector<Int> tmp;
        tmp.reserve(static_cast<size_t>(max_nnz));

        for (size_t col_idx = dim_first[dim]; col_idx < dim_last[dim]; ++col_idx) {
            int target_nnz = 0;
            if (is_nonempty(rng)) {
                target_nnz = 1 + std::min(max_nnz - 1, extra_nnz(rng));
            }
            tmp.clear();
            while (static_cast<int>(tmp.size()) < target_nnz) {
                const Int row_idx = row_dist(rng);
                if (std::find(tmp.begin(), tmp.end(), row_idx) == tmp.end()) {
                    tmp.push_back(row_idx);
                }
            }
            std::sort(tmp.begin(), tmp.end());
            cols[col_idx].assign(tmp.begin(), tmp.end());
        }
    }

    for (size_t dim = 0; dim < dim_first.size(); ++dim) {
        const size_t col_start = dim_first[dim];
        const size_t col_end = dim_last[dim];
        const Matrix<Int> expected = col_to_row_reference<Int>(cols, col_start, col_end, num_rows);

        for (int threads : {1, 8, 16}) {
            for (bool prefer_row_scatter : {false, true}) {
                const Matrix<Int> got = MatrixTraits::col_to_row_format_parallel(
                        cols, threads, col_start, col_end, num_rows, prefer_row_scatter);
                REQUIRE(got == expected);
                REQUIRE(rows_are_sorted(got));
            }
            // 200k rows = 7 bands: the bucket variant crosses band boundaries here
            const Matrix<Int> got_bucket = MatrixTraits::col_to_row_format_bucket(
                    cols, threads, col_start, col_end, num_rows);
            REQUIRE(got_bucket == expected);
            REQUIRE(rows_are_sorted(got_bucket));
        }

        if (dim == 0) {
            for (const auto& row : expected) {
                REQUIRE(row.empty());
            }
        } else {
            const size_t active_lo = dim_first[dim - 1];
            const size_t active_hi = dim_last[dim - 1];
            for (size_t row_idx = 0; row_idx < expected.size(); ++row_idx) {
                if (row_idx < active_lo || row_idx >= active_hi) {
                    REQUIRE(expected[row_idx].empty());
                }
            }
        }
    }
}
