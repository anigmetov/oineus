#include <catch2/catch_test_macros.hpp>

#include <oineus/sparse_matrix.h>

using namespace oineus;

using Int = int;

TEST_CASE("Matrix creation")
{
    {
        SparseMatrix<Int> m(2, 3);
        REQUIRE((m.n_rows() == 2 and m.n_cols() == 3 and m.is_col_zero(0) and m.is_col_zero(1) and m.is_col_zero(2) and m.is_row_zero(0) and m.is_row_zero(1) and m.sanity_check()));
    }

    {
        SparseColumn<Int> col_0 { 0, 1, 2 };
        SparseColumn<Int> col_1 { 0, 1 };
        SparseMatrix<Int>::Data col_data { col_0, col_1 };
        SparseMatrix<Int> m(col_data, 3, true);

        SparseColumn<Int> row_0 { 0, 1 };
        SparseColumn<Int> row_1 { 0, 1 };
        SparseColumn<Int> row_2 { 0 };
        REQUIRE((m.n_rows() == 3 and m.n_cols() == 2 and m.row(0) == row_0 and m.row(1) == row_1 and m.row(2) == row_2 and m.sanity_check()));
    }
}

TEST_CASE("Generic transpose parallel branch returns row format")
{
    std::vector<std::vector<Int>> col_format(45);
    for (Int col_idx = 0; col_idx < static_cast<Int>(col_format.size()); ++col_idx) {
        col_format[col_idx].push_back(col_idx % 4);
        if (col_idx % 3 == 0) {
            col_format[col_idx].push_back(4);
        }
    }

    std::vector<std::vector<Int>> expected(5);
    for (Int col_idx = 0; col_idx < static_cast<Int>(col_format.size()); ++col_idx) {
        for (Int row_idx : col_format[col_idx]) {
            expected[row_idx].push_back(col_idx);
        }
    }

    auto row_format = oineus::transpose(col_format, 5, 2);

    REQUIRE(row_format == expected);
}
