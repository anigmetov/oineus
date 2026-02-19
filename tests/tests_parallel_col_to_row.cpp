#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

using Matrix = std::vector<std::vector<int>>;

Matrix col_to_row_format_serial_reference(const Matrix& col_format, size_t col_start = 0,
        size_t col_end = std::numeric_limits<size_t>::max(), int num_rows = -1)
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

    Matrix row_format(static_cast<size_t>(num_rows));

    for (size_t col_idx = col_start; col_idx < col_end; ++col_idx) {
        for (int row_idx : col_format[col_idx]) {
            row_format[static_cast<size_t>(row_idx)].push_back(static_cast<int>(col_idx));
        }
    }

    return row_format;
}

Matrix col_to_row_format_parallel(const Matrix& col_format, int n_threads, size_t col_start = 0,
        size_t col_end = std::numeric_limits<size_t>::max(), int num_rows = -1)
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

bool rows_are_sorted(const Matrix& rows)
{
    for (const auto& row : rows) {
        if (!std::is_sorted(row.begin(), row.end())) {
            return false;
        }
    }
    return true;
}

Matrix make_sparse_columns(size_t n_cols, int n_rows, int max_nnz_per_col, double nonempty_col_prob,
        uint64_t seed)
{
    Matrix cols(n_cols);

    if (n_rows <= 0 || max_nnz_per_col <= 0) {
        return cols;
    }

    std::mt19937_64 rng(seed);
    std::bernoulli_distribution is_nonempty(nonempty_col_prob);
    std::geometric_distribution<int> extra_nnz(0.7);
    std::uniform_int_distribution<int> row_dist(0, n_rows - 1);

    std::vector<int> tmp;
    tmp.reserve(static_cast<size_t>(max_nnz_per_col));

    for (auto& col : cols) {
        int target_nnz = 0;
        if (is_nonempty(rng)) {
            const int capped_extra = std::min(max_nnz_per_col - 1, extra_nnz(rng));
            target_nnz = 1 + capped_extra;
        }
        tmp.clear();
        while (static_cast<int>(tmp.size()) < target_nnz) {
            const int row_idx = row_dist(rng);
            if (std::find(tmp.begin(), tmp.end(), row_idx) == tmp.end()) {
                tmp.push_back(row_idx);
            }
        }
        std::sort(tmp.begin(), tmp.end());
        col = tmp;
    }

    return cols;
}

Matrix make_sparse_columns(size_t n_cols, int n_rows, int max_nnz_per_col, uint64_t seed)
{
    return make_sparse_columns(n_cols, n_rows, max_nnz_per_col, 0.15, seed);
}

} // namespace

TEST_CASE("Parallel col->row: handles empty and trivial inputs")
{
    {
        Matrix cols;
        REQUIRE(col_to_row_format_parallel(cols, 4).empty());
    }

    {
        Matrix cols(10);
        const auto got = col_to_row_format_parallel(cols, 8);
        REQUIRE(got.empty());
    }

    {
        Matrix cols{{0, 2, 4}, {1, 2}, {}, {0, 3, 4}};
        const auto expected = col_to_row_format_serial_reference(cols);
        const auto got = col_to_row_format_parallel(cols, 3);
        REQUIRE(got == expected);
        REQUIRE(rows_are_sorted(got));
    }
}

TEST_CASE("Parallel col->row: randomized correctness across thread counts")
{
    std::mt19937_64 rng(1234567);
    std::uniform_int_distribution<int> cols_dist(0, 300);
    std::uniform_int_distribution<int> rows_dist(0, 400);
    std::uniform_int_distribution<int> nnz_dist(2, 4);
    std::uniform_real_distribution<double> nonempty_prob_dist(0.03, 0.20);

    for (int trial = 0; trial < 80; ++trial) {
        const size_t n_cols = static_cast<size_t>(cols_dist(rng));
        const int n_rows = rows_dist(rng);
        const int max_nnz = nnz_dist(rng);
        const double nonempty_prob = nonempty_prob_dist(rng);

        Matrix cols = make_sparse_columns(
                n_cols, n_rows, max_nnz, nonempty_prob, 1000u + static_cast<uint64_t>(trial));
        const Matrix expected = col_to_row_format_serial_reference(cols);

        for (int threads : {1, 2, 3, 4, 8, 16}) {
            const Matrix got = col_to_row_format_parallel(cols, threads);
            REQUIRE(got == expected);
            REQUIRE(rows_are_sorted(got));
        }
    }
}

TEST_CASE("Parallel col->row: 100x100 sanity and speed")
{
    const Matrix cols = make_sparse_columns(100, 100, 3, 0.10, 20260219);
    const Matrix expected = col_to_row_format_serial_reference(cols);

    const auto t0 = std::chrono::steady_clock::now();
    for (int threads : {1, 2, 4, 8, 16}) {
        const Matrix got = col_to_row_format_parallel(cols, threads);
        REQUIRE(got == expected);
        REQUIRE(rows_are_sorted(got));
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    REQUIRE(elapsed_sec < 2.0);
}

TEST_CASE("Parallel col->row: medium sparse stress")
{
    const size_t n_cols = 10000;
    const int n_rows = 12000;
    const int max_nnz = 3;

    const Matrix cols = make_sparse_columns(n_cols, n_rows, max_nnz, 0.08, 42);
    const Matrix expected = col_to_row_format_serial_reference(cols);

    for (int threads : {1, 2, 4, 8, 16}) {
        const Matrix got = col_to_row_format_parallel(cols, threads);
        REQUIRE(got == expected);
        REQUIRE(rows_are_sorted(got));
    }
}

TEST_CASE("Parallel col->row: large sparse case up to 200k columns")
{
    const size_t n_cols = 200000;
    const int n_rows = 220000;
    const int max_nnz = 3;

    const Matrix cols = make_sparse_columns(n_cols, n_rows, max_nnz, 0.05, 987654321ull);
    const Matrix expected = col_to_row_format_serial_reference(cols);

    for (int threads : {1, 8, 16}) {
        const Matrix got = col_to_row_format_parallel(cols, threads);
        REQUIRE(got == expected);
        REQUIRE(rows_are_sorted(got));
    }
}

TEST_CASE("Parallel col->row: subrange semantics match serial reference")
{
    const Matrix cols{
            {},
            {0, 2, 5},
            {1},
            {0, 4},
            {},
            {3, 5},
    };

    const size_t col_start = 1;
    const size_t col_end = 5;
    const int num_rows = 6;

    const Matrix expected = col_to_row_format_serial_reference(cols, col_start, col_end, num_rows);
    for (int threads : {1, 2, 4, 8}) {
        const Matrix got = col_to_row_format_parallel(cols, threads, col_start, col_end, num_rows);
        REQUIRE(got == expected);
        REQUIRE(rows_are_sorted(got));
    }

    const Matrix expected_clamped = col_to_row_format_serial_reference(cols, 2, 1000, num_rows);
    const Matrix got_clamped = col_to_row_format_parallel(cols, 4, 2, 1000, num_rows);
    REQUIRE(got_clamped == expected_clamped);
    REQUIRE(rows_are_sorted(got_clamped));
}

TEST_CASE("Parallel col->row: boundary-like 200k matrix per-dimension block conversion")
{
    const size_t n_cols = 200000;
    const int num_rows = static_cast<int>(n_cols);
    const int max_nnz = 3;

    // Exclusive ranges for dimensions 0..3.
    const std::vector<size_t> dim_first{0, 50000, 110000, 160000};
    const std::vector<size_t> dim_last{50000, 110000, 160000, 200000};

    Matrix cols(n_cols);
    std::mt19937_64 rng(20260219);
    std::bernoulli_distribution is_nonempty(0.05);
    std::geometric_distribution<int> extra_nnz(0.7);

    // dim 0: empty columns (boundary of 0-cells)
    for (size_t col_idx = dim_first[0]; col_idx < dim_last[0]; ++col_idx) {
        cols[col_idx].clear();
    }

    // dim d>0: nonzero rows only in previous dimension range.
    for (size_t dim = 1; dim < dim_first.size(); ++dim) {
        const int row_lo = static_cast<int>(dim_first[dim - 1]);
        const int row_hi = static_cast<int>(dim_last[dim - 1] - 1);
        std::uniform_int_distribution<int> row_dist(row_lo, row_hi);

        std::vector<int> tmp;
        tmp.reserve(static_cast<size_t>(max_nnz));

        for (size_t col_idx = dim_first[dim]; col_idx < dim_last[dim]; ++col_idx) {
            int target_nnz = 0;
            if (is_nonempty(rng)) {
                target_nnz = 1 + std::min(max_nnz - 1, extra_nnz(rng));
            }
            tmp.clear();
            while (static_cast<int>(tmp.size()) < target_nnz) {
                const int row_idx = row_dist(rng);
                if (std::find(tmp.begin(), tmp.end(), row_idx) == tmp.end()) {
                    tmp.push_back(row_idx);
                }
            }
            std::sort(tmp.begin(), tmp.end());
            cols[col_idx] = tmp;
        }
    }

    for (size_t dim = 0; dim < dim_first.size(); ++dim) {
        const size_t col_start = dim_first[dim];
        const size_t col_end = dim_last[dim];
        const Matrix expected = col_to_row_format_serial_reference(cols, col_start, col_end, num_rows);

        for (int threads : {1, 8, 16}) {
            const Matrix got = col_to_row_format_parallel(cols, threads, col_start, col_end, num_rows);
            REQUIRE(got == expected);
            REQUIRE(rows_are_sorted(got));
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
