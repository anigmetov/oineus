// Unit tests for U-computation primitives:
// - compute_u_column_1 (column-form V U = I solve, used by
//   compute_u_from_v_1).
// - compute_u_row_bounded, compute_full_u_rows, compute_partial_u_rows
//   (row-form U^T V^T = I solves, used by the row_partial u_strategy).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <oineus/oineus.h>

using dim_type = size_t;

// Small helper: build a random freudenthal grid filtration of a tiny
// 2-D grid and reduce it. Reductions vary by the params we feed: with
// or without clearing, with or without restore_elz, with or without
// in-band U.
namespace {

template<class Int, class Real>
oineus::Filtration<oineus::Simplex<Int>, Real>
make_test_filtration(unsigned nx, unsigned ny, unsigned seed = 7)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    using GridT = oineus::Grid<Int, Real, 2>;
    typename GridT::GridPoint dims{nx, ny};
    std::vector<Real> data(nx * ny);
    for (auto& v : data) v = dist(rng);
    GridT grid(dims, /*wrap=*/false, data.data(), GridT::DataLocation::VERTEX);
    return grid.freudenthal_filtration(/*top_d=*/2, /*negate=*/false, /*n_threads=*/1);
}

template<class Int, class Real, class Fil>
oineus::VRUDecomposition<Int>
reduce_with_params(const Fil& fil, bool clearing, bool compute_u,
                   bool restore_elz, int n_threads = 1)
{
    oineus::VRUDecomposition<Int> decmp(fil, /*dualize=*/false);
    oineus::Params params;
    params.compute_v = true;
    params.compute_u = compute_u;
    params.clearing_opt = clearing;
    if (restore_elz)
        for (oineus::dim_type d = 0; d < static_cast<oineus::dim_type>(decmp.n_dims()); ++d)
            params.dims_to_restore_elz.push_back(d);
    params.n_threads = n_threads;
    decmp.reduce(params);
    return decmp;
}

template<class Int, class Real, class Fil>
oineus::VRUDecomposition<Int>
reduce_with_params_dualize(const Fil& fil, bool dualize, bool clearing,
                           bool compute_u, bool restore_elz,
                           int n_threads = 1)
{
    oineus::VRUDecomposition<Int> decmp(fil, dualize);
    oineus::Params params;
    params.compute_v = true;
    params.compute_u = compute_u;
    params.clearing_opt = clearing;
    if (restore_elz)
        for (oineus::dim_type d = 0; d < static_cast<oineus::dim_type>(decmp.n_dims()); ++d)
            params.dims_to_restore_elz.push_back(d);
    params.n_threads = n_threads;
    decmp.reduce(params);
    return decmp;
}

}  // namespace



TEST_CASE("compute_u_column_1 with restore_elz matches the in-band U")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(5, 5);

    auto decmp_ref = reduce_with_params<Int, Real>(fil, /*clearing=*/false,
                                                   /*compute_u=*/true,
                                                   /*restore_elz=*/false);

    auto decmp = reduce_with_params<Int, Real>(fil, /*clearing=*/true,
                                               /*compute_u=*/false,
                                               /*restore_elz=*/true);

    for (size_t c = 0; c < decmp.size(); ++c) {
        // match the at-rest column type returned by compute_u_column_1 so the
        // REQUIRE(solved == ref_col) compares like with like
        typename std::decay_t<decltype(decmp)>::IntSparseColumn ref_col;
        for (size_t r = 0; r < decmp_ref.u_data_t.size(); ++r) {
            const auto& row = decmp_ref.u_data_t[r];
            if (std::find(row.begin(), row.end(), static_cast<Int>(c)) != row.end()) {
                ref_col.push_back(static_cast<Int>(r));
            }
        }
        std::sort(ref_col.begin(), ref_col.end());

        auto solved = decmp.compute_u_column_1(c);
        REQUIRE(solved == ref_col);
    }
}


// ============================================================================
// Row-form U primitive tests.
// ============================================================================


// Build a value_at lambda for a Decomposition: maps matrix-column index to
// the cell's filtration value, respecting dualize.
template<class Int, class Real, class Fil>
auto make_value_at(const Fil& fil, bool dualize)
{
    return [&fil, dualize](Int matrix_idx) -> Real {
        return fil.get_cell_value(
            fil.index_in_filtration(static_cast<size_t>(matrix_idx), dualize));
    };
}


TEST_CASE("compute_u_row_bounded full row matches in-band U row, hom")
{
    using Int = long;
    using Real = double;
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    // Reference: in-band U on hom side.
    auto decmp_ref = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/false,
            /*compute_u=*/true, /*restore_elz=*/false);

    // Phase-4 setup: clearing+restore on hom side, no in-band U.
    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/false);
    auto never_stop = [](Real, Real) { return false; };

    // dim 1 = edges
    const dim_type dim = 1;
    const size_t cstart = decmp.range_start_(dim);
    const size_t cend = decmp.range_end_(dim);
    REQUIRE(cend > cstart);

    auto vt_data = MatrixTraits::col_to_row_format_parallel(
            decmp.v_data, /*n_threads=*/1, cstart, cend,
            static_cast<Int>(decmp.v_data.size()));

    for (size_t r = cstart; r < cend; ++r) {
        auto row = decmp.compute_u_row_bounded(
                r, vt_data, std::numeric_limits<Real>::max(),
                value_at, never_stop);
        REQUIRE(row == decmp_ref.u_data_t[r]);
    }
}


TEST_CASE("compute_u_row_bounded full row matches in-band U row, coh")
{
    using Int = long;
    using Real = double;
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    auto decmp_ref = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/true, /*clearing=*/false,
            /*compute_u=*/true, /*restore_elz=*/false);

    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/true, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/true);
    auto never_stop = [](Real, Real) { return false; };

    // For coh, dim_first/dim_last are reversed; just walk every dim.
    auto vt_data_full = MatrixTraits::col_to_row_format_parallel(
            decmp.v_data, /*n_threads=*/1, 0, decmp.v_data.size(),
            static_cast<Int>(decmp.v_data.size()));

    for (size_t r = 0; r < decmp.size(); ++r) {
        auto row = decmp.compute_u_row_bounded(
                r, vt_data_full, std::numeric_limits<Real>::max(),
                value_at, never_stop);
        REQUIRE(row == decmp_ref.u_data_t[r]);
    }
}


TEST_CASE("compute_u_row_bounded with bound is a prefix of the full row, hom")
{
    using Int = long;
    using Real = double;
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/false);
    auto never_stop = [](Real, Real) { return false; };
    auto stop_above = [](Real piv_value, Real value_bound) {
        return piv_value > value_bound;
    };

    const dim_type dim = 1;
    const size_t cstart = decmp.range_start_(dim);
    const size_t cend = decmp.range_end_(dim);
    auto vt_data = MatrixTraits::col_to_row_format_parallel(
            decmp.v_data, 1, cstart, cend,
            static_cast<Int>(decmp.v_data.size()));

    for (size_t r = cstart; r < cend; ++r) {
        auto unbounded = decmp.compute_u_row_bounded(
                r, vt_data, std::numeric_limits<Real>::max(),
                value_at, never_stop);
        if (unbounded.empty()) continue;

        // Pick a bound = median value among the unbounded entries.
        std::vector<Real> values;
        values.reserve(unbounded.size());
        for (auto idx : unbounded) values.push_back(value_at(idx));
        std::sort(values.begin(), values.end());
        Real bound = values[values.size() / 2];

        auto bounded = decmp.compute_u_row_bounded(
                r, vt_data, bound, value_at, stop_above);

        // Every bounded entry is in the unbounded result.
        for (auto idx : bounded) {
            REQUIRE(std::find(unbounded.begin(), unbounded.end(), idx)
                    != unbounded.end());
        }
        // Every bounded entry has value <= bound, modulo the
        // diagonal-element fallback (bounded == {r}).
        if (!(bounded.size() == 1 && bounded.front() == static_cast<Int>(r))) {
            for (auto idx : bounded) {
                REQUIRE(value_at(idx) <= bound);
            }
        }
    }
}


TEST_CASE("compute_u_row_bounded with bound is a prefix of the full row, coh")
{
    using Int = long;
    using Real = double;
    using MatrixTraits = oineus::SimpleSparseMatrixTraits<Int, 2>;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/true, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/true);
    auto never_stop = [](Real, Real) { return false; };
    // For coh: pivot values DECREASE in matrix order, so the truncation
    // direction is "stop when piv falls below bound".
    auto stop_below = [](Real piv_value, Real value_bound) {
        return piv_value < value_bound;
    };

    auto vt_data = MatrixTraits::col_to_row_format_parallel(
            decmp.v_data, 1, 0, decmp.v_data.size(),
            static_cast<Int>(decmp.v_data.size()));

    for (size_t r = 0; r < decmp.size(); ++r) {
        auto unbounded = decmp.compute_u_row_bounded(
                r, vt_data, -std::numeric_limits<Real>::max(),
                value_at, never_stop);
        if (unbounded.empty()) continue;

        std::vector<Real> values;
        values.reserve(unbounded.size());
        for (auto idx : unbounded) values.push_back(value_at(idx));
        std::sort(values.begin(), values.end());
        Real bound = values[values.size() / 2];

        auto bounded = decmp.compute_u_row_bounded(
                r, vt_data, bound, value_at, stop_below);

        for (auto idx : bounded) {
            REQUIRE(std::find(unbounded.begin(), unbounded.end(), idx)
                    != unbounded.end());
        }
        if (!(bounded.size() == 1 && bounded.front() == static_cast<Int>(r))) {
            for (auto idx : bounded) {
                REQUIRE(value_at(idx) >= bound);
            }
        }
    }
}


TEST_CASE("compute_full_u_rows matches compute_u_from_v_1 row-by-row, hom")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(5, 5);

    auto decmp_ref = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);
    decmp_ref.compute_u_from_v_1(/*dim=*/1, /*n_threads=*/1);

    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/false);
    decmp.compute_full_u_rows<Real>(/*dim=*/1, value_at, /*n_threads=*/1);

    REQUIRE(decmp.u_data_t.size() == decmp_ref.u_data_t.size());
    for (size_t r = 0; r < decmp.u_data_t.size(); ++r) {
        REQUIRE(decmp.u_data_t[r] == decmp_ref.u_data_t[r]);
    }
}


TEST_CASE("compute_full_u_rows matches compute_u_from_v_1 row-by-row, coh")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(5, 5);

    auto decmp_ref = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/true, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);
    decmp_ref.compute_u_from_v_1(/*dim=*/1, /*n_threads=*/1);

    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/true, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/true);
    decmp.compute_full_u_rows<Real>(/*dim=*/1, value_at, /*n_threads=*/1);

    REQUIRE(decmp.u_data_t.size() == decmp_ref.u_data_t.size());
    for (size_t r = 0; r < decmp.u_data_t.size(); ++r) {
        REQUIRE(decmp.u_data_t[r] == decmp_ref.u_data_t[r]);
    }
}


TEST_CASE("compute_partial_u_rows writes only requested rows, hom")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    // Reference: in-band U.
    auto decmp_ref = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/false,
            /*compute_u=*/true, /*restore_elz=*/false);

    auto decmp = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/true,
            /*compute_u=*/false, /*restore_elz=*/true);

    const dim_type dim = 1;
    const size_t cstart = decmp.range_start_(dim);
    const size_t cend = decmp.range_end_(dim);
    REQUIRE(cend - cstart >= 4);

    // Pick every other row.
    std::vector<size_t> rows;
    std::vector<Real> bounds;
    for (size_t r = cstart; r < cend; r += 2) {
        rows.push_back(r);
        bounds.push_back(std::numeric_limits<Real>::max());
    }

    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/false);
    auto never_stop = [](Real, Real) { return false; };

    decmp.compute_partial_u_rows(rows, bounds, dim, value_at, never_stop, 1);

    std::set<size_t> rows_set(rows.begin(), rows.end());
    for (size_t r = 0; r < decmp.u_data_t.size(); ++r) {
        if (rows_set.count(r)) {
            REQUIRE(decmp.u_data_t[r] == decmp_ref.u_data_t[r]);
        } else {
            // Untouched rows stay empty.
            REQUIRE(decmp.u_data_t[r].empty());
        }
    }
}


TEST_CASE("compute_partial_u_rows is deterministic across thread counts")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(5, 5);

    auto run = [&fil](size_t n_threads) {
        auto decmp = reduce_with_params_dualize<Int, Real>(
                fil, /*dualize=*/false, /*clearing=*/true,
                /*compute_u=*/false, /*restore_elz=*/true);
        const dim_type dim = 1;
        const size_t cstart = decmp.range_start_(dim);
        const size_t cend = decmp.range_end_(dim);
        std::vector<size_t> rows;
        std::vector<Real> bounds;
        for (size_t r = cstart; r < cend; ++r) {
            rows.push_back(r);
            bounds.push_back(std::numeric_limits<Real>::max());
        }
        auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/false);
        auto never_stop = [](Real, Real) { return false; };
        decmp.compute_partial_u_rows(rows, bounds, dim, value_at,
                                     never_stop, n_threads);
        return decmp.u_data_t;
    };

    auto u1 = run(1);
    auto u4 = run(4);
    REQUIRE(u1.size() == u4.size());
    for (size_t r = 0; r < u1.size(); ++r) {
        REQUIRE(u1[r] == u4[r]);
    }
}
