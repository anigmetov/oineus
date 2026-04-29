// Phase-3 unit tests for the bounded variants of compute_u_column
// (Algorithm 3) and compute_u_column_1 (Algorithm 4).

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
    params.restore_elz = restore_elz;
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
    params.restore_elz = restore_elz;
    params.n_threads = n_threads;
    decmp.reduce(params);
    return decmp;
}

}  // namespace


TEST_CASE("compute_u_column_bounded with no bound matches compute_u_column")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    // Reference: Phase-2-style in-band U, no clearing.
    auto decmp_ref = reduce_with_params<Int, Real>(fil, /*clearing=*/false,
                                                   /*compute_u=*/true,
                                                   /*restore_elz=*/false);

    // Phase-3-style: clearing on, U computed via Algorithm 3 post-pass.
    auto decmp = reduce_with_params<Int, Real>(fil, /*clearing=*/true,
                                               /*compute_u=*/false,
                                               /*restore_elz=*/true);

    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };
    auto never_stop = [](Real, Real) { return false; };

    for (size_t c = 0; c < decmp.size(); ++c) {
        auto bounded = decmp.compute_u_column_bounded(
            c, /*value_bound=*/std::numeric_limits<Real>::max(),
            value_at, never_stop);
        auto unbounded = decmp.compute_u_column(c);
        REQUIRE(bounded == unbounded);
    }
}


TEST_CASE("compute_u_column_1_bounded with no bound matches compute_u_column_1")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    auto decmp = reduce_with_params<Int, Real>(fil, /*clearing=*/true,
                                               /*compute_u=*/false,
                                               /*restore_elz=*/true);

    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };
    auto never_stop = [](Real, Real) { return false; };

    for (size_t c = 0; c < decmp.size(); ++c) {
        auto bounded = decmp.compute_u_column_1_bounded(
            c, /*value_bound=*/std::numeric_limits<Real>::max(),
            value_at, never_stop);
        auto unbounded = decmp.compute_u_column_1(c);
        REQUIRE(bounded == unbounded);
    }
}


TEST_CASE("compute_u_column_1_bounded with strict bound is a prefix of the full column")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    auto decmp = reduce_with_params<Int, Real>(fil, /*clearing=*/true,
                                               /*compute_u=*/false,
                                               /*restore_elz=*/true);

    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };
    // For Algorithm 4 the visited pivots strictly decrease in index;
    // cmp_op = "value strictly below bound" matches an
    // increase_birth-style early exit. We assert prefix-equality on
    // the iteration order: the bounded result, before sorting, is the
    // prefix of the unbounded iteration order. Both functions sort
    // their result before returning, so we compare sorted prefixes
    // via the value bound.
    for (size_t c = 0; c < decmp.size(); ++c) {
        auto unbounded = decmp.compute_u_column_1(c);
        if (unbounded.empty()) continue;
        // Pick a bound roughly halfway through the unbounded entries.
        Real bound;
        {
            std::vector<Real> values;
            values.reserve(unbounded.size());
            for (auto idx : unbounded) values.push_back(value_at(idx));
            std::sort(values.begin(), values.end());
            bound = values[values.size() / 2];
        }
        auto stop_below = [](Real piv_value, Real value_bound) {
            return piv_value < value_bound;
        };
        auto bounded = decmp.compute_u_column_1_bounded(
            c, bound, value_at, stop_below);
        // Every entry in `bounded` must also be in `unbounded`.
        for (auto idx : bounded) {
            REQUIRE(std::find(unbounded.begin(), unbounded.end(), idx)
                    != unbounded.end());
        }
        // Every entry in `bounded` (apart from the diagonal-element
        // fallback when bounded is empty otherwise) must satisfy
        // value(idx) >= bound.
        // Diagonal-element fallback edge case: bounded may consist of
        // {c} alone if the very first iteration's pivot was below the
        // bound; in that case we don't enforce the value bound.
        if (!(bounded.size() == 1 && bounded.front() == static_cast<Int>(c))) {
            for (auto idx : bounded) {
                REQUIRE(value_at(idx) >= bound);
            }
        }
    }
}


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
        std::vector<Int> ref_col;
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


TEST_CASE("compute_partial_u_from_v_1 full cols matches compute_u_from_v_1")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    // Reference: post-pass U from compute_u_from_v_1 over the full
    // dim 1 (Phase-2 / clearing-on, restore-on, V-only style).
    auto decmp_ref = reduce_with_params<Int, Real>(fil, true, false, true);
    decmp_ref.compute_u_from_v_1(/*dim=*/1, /*n_threads=*/1);

    // Partial driver with cols spanning the whole dim 1, never-stop bound.
    auto decmp = reduce_with_params<Int, Real>(fil, true, false, true);
    const size_t col_start = decmp.range_start_(1);
    const size_t col_end = decmp.range_end_(1);
    REQUIRE(col_end > col_start);
    std::vector<size_t> cols;
    std::vector<Real> bounds;
    for (size_t c = col_start; c < col_end; ++c) {
        cols.push_back(c);
        bounds.push_back(std::numeric_limits<Real>::max());
    }
    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };
    auto never_stop = [](Real, Real) { return false; };
    decmp.compute_partial_u_from_v_1(cols, bounds, value_at, never_stop, 1);

    REQUIRE(decmp.u_data_t.size() == decmp_ref.u_data_t.size());
    for (size_t r = 0; r < decmp.u_data_t.size(); ++r) {
        REQUIRE(decmp.u_data_t[r] == decmp_ref.u_data_t[r]);
    }
}


TEST_CASE("compute_partial_u_from_v full cols matches compute_u_from_v")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    auto decmp_ref = reduce_with_params<Int, Real>(fil, true, false, true);
    decmp_ref.compute_u_from_v(/*dim=*/1, /*n_threads=*/1);

    auto decmp = reduce_with_params<Int, Real>(fil, true, false, true);
    const size_t col_start = decmp.range_start_(1);
    const size_t col_end = decmp.range_end_(1);
    REQUIRE(col_end > col_start);
    std::vector<size_t> cols;
    std::vector<Real> bounds;
    for (size_t c = col_start; c < col_end; ++c) {
        cols.push_back(c);
        bounds.push_back(std::numeric_limits<Real>::max());
    }
    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };
    auto never_stop = [](Real, Real) { return false; };
    decmp.compute_partial_u_from_v(cols, bounds, value_at, never_stop, 1);

    REQUIRE(decmp.u_data_t.size() == decmp_ref.u_data_t.size());
    for (size_t r = 0; r < decmp.u_data_t.size(); ++r) {
        REQUIRE(decmp.u_data_t[r] == decmp_ref.u_data_t[r]);
    }
}


TEST_CASE("compute_partial_u_from_v_1 sparse cols subset matches in-band U restriction")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(4, 4);

    // In-band U via clearing-off / compute_u=true.
    auto decmp_ref = reduce_with_params<Int, Real>(fil, false, true, false);

    auto decmp = reduce_with_params<Int, Real>(fil, true, false, true);
    const size_t col_start = decmp.range_start_(1);
    const size_t col_end = decmp.range_end_(1);
    REQUIRE(col_end - col_start >= 2);

    // Pick every other column in dim 1.
    std::vector<size_t> cols;
    std::vector<Real> bounds;
    for (size_t c = col_start; c < col_end; c += 2) {
        cols.push_back(c);
        bounds.push_back(std::numeric_limits<Real>::max());
    }
    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };
    auto never_stop = [](Real, Real) { return false; };
    decmp.compute_partial_u_from_v_1(cols, bounds, value_at, never_stop, 1);

    std::set<Int> cols_set;
    for (auto c : cols) cols_set.insert(static_cast<Int>(c));

    // For each row r in dim 1: u_data_t[r] should equal
    // {c in cols : c in inband_u_data_t[r]}.
    for (size_t r = col_start; r < col_end; ++r) {
        std::vector<Int> filtered_ref;
        for (auto c : decmp_ref.u_data_t[r]) {
            if (cols_set.count(c)) {
                filtered_ref.push_back(c);
            }
        }
        std::sort(filtered_ref.begin(), filtered_ref.end());
        auto out = decmp.u_data_t[r];
        std::sort(out.begin(), out.end());
        REQUIRE(out == filtered_ref);
    }
}


TEST_CASE("compute_partial_u_from_v_1 with value bound is a prefix of the in-band U")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(5, 5);

    auto decmp_ref = reduce_with_params<Int, Real>(fil, false, true, false);

    auto decmp = reduce_with_params<Int, Real>(fil, true, false, true);
    const size_t col_start = decmp.range_start_(1);
    const size_t col_end = decmp.range_end_(1);
    REQUIRE(col_end - col_start >= 4);

    auto value_at = [&fil](Int idx) -> Real {
        return fil.get_cell_value(static_cast<size_t>(idx));
    };

    // Per-column bound = median value of dim-1 cells. Stop-below
    // truncates the residual loop the first time pivot value drops
    // below the median.
    std::vector<Real> all_vals;
    for (size_t c = col_start; c < col_end; ++c) {
        all_vals.push_back(value_at(static_cast<Int>(c)));
    }
    std::sort(all_vals.begin(), all_vals.end());
    Real median = all_vals[all_vals.size() / 2];

    std::vector<size_t> cols;
    std::vector<Real> bounds;
    for (size_t c = col_start; c < col_end; c += 2) {
        cols.push_back(c);
        bounds.push_back(median);
    }
    auto stop_below = [](Real piv_value, Real value_bound) {
        return piv_value < value_bound;
    };
    decmp.compute_partial_u_from_v_1(cols, bounds, value_at, stop_below, 1);

    std::map<size_t, Real> col_bound;
    for (size_t i = 0; i < cols.size(); ++i) col_bound[cols[i]] = bounds[i];

    // For each row r, every c in u_data_t[r] must:
    //   - be in cols (no spurious entries),
    //   - appear in the in-band reference at the same (r, c)
    //     position (no false positives),
    //   - either match the diagonal (r == c, the diagonal-element
    //     fallback) or satisfy the value bound (value(r) >= bounds[c]).
    for (size_t r = 0; r < decmp.u_data_t.size(); ++r) {
        for (auto c : decmp.u_data_t[r]) {
            REQUIRE(col_bound.count(static_cast<size_t>(c)));
            const auto& ref_row = decmp_ref.u_data_t[r];
            REQUIRE(std::find(ref_row.begin(), ref_row.end(), c) != ref_row.end());
            if (static_cast<size_t>(c) != r) {
                REQUIRE(value_at(static_cast<Int>(r))
                        >= col_bound[static_cast<size_t>(c)]);
            }
        }
    }
}


TEST_CASE("compute_partial_u_from_v_1 is deterministic across thread counts")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(5, 5);

    auto run_with_threads = [&fil](size_t n_threads) {
        auto decmp = reduce_with_params<Int, Real>(fil, true, false, true);
        const size_t col_start = decmp.range_start_(1);
        const size_t col_end = decmp.range_end_(1);
        std::vector<size_t> cols;
        std::vector<Real> bounds;
        for (size_t c = col_start; c < col_end; ++c) {
            cols.push_back(c);
            bounds.push_back(std::numeric_limits<Real>::max());
        }
        auto value_at = [&fil](Int idx) -> Real {
            return fil.get_cell_value(static_cast<size_t>(idx));
        };
        auto never_stop = [](Real, Real) { return false; };
        decmp.compute_partial_u_from_v_1(cols, bounds, value_at, never_stop,
                                         n_threads);
        return decmp.u_data_t;
    };

    auto u1 = run_with_threads(1);
    auto u4 = run_with_threads(4);

    REQUIRE(u1.size() == u4.size());
    for (size_t r = 0; r < u1.size(); ++r) {
        REQUIRE(u1[r] == u4[r]);
    }
}


// ============================================================================
// Phase-4: row-form U primitive tests.
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
