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
    oineus::ReductionParams params;
    params.compute_v = true;
    params.compute_u = compute_u;
    params.use_clearing = clearing;
    if (restore_elz)
        for (oineus::dim_type d = 0; d < static_cast<oineus::dim_type>(decmp.n_dims()); ++d)
            params.advanced.dims_to_restore_elz.push_back(d);
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
    oineus::ReductionParams params;
    params.compute_v = true;
    params.compute_u = compute_u;
    params.use_clearing = clearing;
    if (restore_elz)
        for (oineus::dim_type d = 0; d < static_cast<oineus::dim_type>(decmp.n_dims()); ++d)
            params.advanced.dims_to_restore_elz.push_back(d);
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


// The bauer phase (recovering cleared V columns) is split out of restore_elz in
// the timings, but must still be part of the comparable reduction_total. Guard
// the formula so it cannot be silently dropped from the total.
TEST_CASE("ReductionTimings.reduction_total includes the bauer phase")
{
    oineus::ReductionTimings t;
    t.prepare = 1.0; t.reduce = 2.0; t.bauer = 4.0;
    t.restore_elz = 8.0; t.copy_back = 16.0; t.copy_pivots = 32.0;
    REQUIRE(t.reduction_total() == 63.0);  // fails if bauer (4) is dropped
}


// The benchmark drives three full-inversion strategies on the SAME reduced
// (clearing + restore_elz) decomposition and must get the same U from each,
// while u_timings records the per-phase split. This guards both invariants:
//   - compute_u_from_v   (RUD, R u_c = D_c, column-form)   -> col_solve + col_to_row
//   - compute_u_from_v_1 (VUI, V u_c = e_c, column-form)   -> col_solve + col_to_row
//   - compute_full_u_rows (V^T U^T = Id, row-form)         -> transpose_v + row_solve
// all equal the in-band U over the dim-1 row block.
TEST_CASE("full-inversion strategies agree on U and populate u_timings")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(6, 6);

    // In-band reference U (clearing off, serial).
    auto decmp_ref = reduce_with_params_dualize<Int, Real>(
            fil, /*dualize=*/false, /*clearing=*/false,
            /*compute_u=*/true, /*restore_elz=*/false);

    const dim_type dim = 1;

    auto fresh = [&fil]() {
        // The benchmark's reduction config: parallel clearing + restore_elz.
        return reduce_with_params_dualize<Int, Real>(
                fil, /*dualize=*/false, /*clearing=*/true,
                /*compute_u=*/false, /*restore_elz=*/true, /*n_threads=*/4);
    };
    auto value_at = make_value_at<Int, Real>(fil, /*dualize=*/false);

    // The dim-1 row block; the three strategies only touch these rows.
    auto rng = fresh();
    const size_t cstart = rng.range_start_(dim);
    const size_t cend = rng.range_end_(dim);
    REQUIRE(cend > cstart);

    auto check_dim1_rows = [&](const auto& decmp) {
        REQUIRE(decmp.u_data_t.size() == decmp_ref.u_data_t.size());
        for (size_t r = cstart; r < cend; ++r)
            REQUIRE(decmp.u_data_t[r] == decmp_ref.u_data_t[r]);
    };

    // RUD: R u_c = D_c (needs d_data, kept by the ctor+reduce path).
    auto decmp_rud = fresh();
    decmp_rud.compute_u_from_v(dim, /*n_threads=*/4);
    check_dim1_rows(decmp_rud);
    REQUIRE(decmp_rud.u_timings_.col_solve > 0.0);
    REQUIRE(decmp_rud.u_timings_.transpose_v == 0.0);
    REQUIRE(decmp_rud.u_timings_.row_solve == 0.0);

    // VUI: V u_c = e_c.
    auto decmp_vui = fresh();
    decmp_vui.compute_u_from_v_1(dim, /*n_threads=*/4);
    check_dim1_rows(decmp_vui);
    REQUIRE(decmp_vui.u_timings_.col_solve > 0.0);
    REQUIRE(decmp_vui.u_timings_.transpose_v == 0.0);

    // VtUt: V^T U^T = Id (row-form).
    auto decmp_vtut = fresh();
    decmp_vtut.compute_full_u_rows<Real>(dim, value_at, /*n_threads=*/4);
    check_dim1_rows(decmp_vtut);
    REQUIRE(decmp_vtut.u_timings_.transpose_v > 0.0);
    REQUIRE(decmp_vtut.u_timings_.row_solve > 0.0);
    REQUIRE(decmp_vtut.u_timings_.col_solve == 0.0);
}


// The U-solve must honor the reduction's col_repr: whatever working-column
// data structure the reduction ran with, the U it computes must be identical.
// These cross-check every supported residual against the BitTree default.
namespace {

template<class Int, class Real, class Fil>
oineus::VRUDecomposition<Int>
reduce_col_repr(const Fil& fil, bool dualize, oineus::ColumnRepr cr,
                int n_threads = 1)
{
    oineus::VRUDecomposition<Int> decmp(fil, dualize);
    oineus::ReductionParams params;
    params.compute_v = true;
    params.use_clearing = true;
    params.advanced.col_repr = cr;
    for (oineus::dim_type d = 0;
         d < static_cast<oineus::dim_type>(decmp.n_dims()); ++d)
        params.advanced.dims_to_restore_elz.push_back(d);
    params.n_threads = n_threads;
    decmp.reduce(params);
    return decmp;
}

}  // namespace


TEST_CASE("compute_u_from_v / _1 honor col_repr (column form, all four)")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(6, 6);
    const dim_type dim = 1;

    for (bool dualize : {false, true}) {
        auto ref = reduce_col_repr<Int, Real>(fil, dualize, oineus::ColumnRepr::BitTree);
        ref.compute_u_from_v(dim, /*n_threads=*/1);

        for (auto cr : {oineus::ColumnRepr::Set, oineus::ColumnRepr::Heap,
                        oineus::ColumnRepr::Full, oineus::ColumnRepr::BitTree}) {
            // RUD (R u_c = D_c) column form.
            auto d1 = reduce_col_repr<Int, Real>(fil, dualize, cr);
            d1.compute_u_from_v(dim, /*n_threads=*/4);
            REQUIRE(d1.u_data_t.size() == ref.u_data_t.size());
            for (size_t r = 0; r < d1.u_data_t.size(); ++r)
                REQUIRE(d1.u_data_t[r] == ref.u_data_t[r]);

            // VUI (V u_c = e_c) column form.
            auto d2 = reduce_col_repr<Int, Real>(fil, dualize, cr);
            d2.compute_u_from_v_1(dim, /*n_threads=*/4);
            REQUIRE(d2.u_data_t.size() == ref.u_data_t.size());
            for (size_t r = 0; r < d2.u_data_t.size(); ++r)
                REQUIRE(d2.u_data_t[r] == ref.u_data_t[r]);
        }
    }
}


TEST_CASE("compute_full_u_rows honors col_repr (row form: Set/Full/BitTree)")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(6, 6);
    const dim_type dim = 1;

    for (bool dualize : {false, true}) {
        auto value_at = make_value_at<Int, Real>(fil, dualize);
        auto ref = reduce_col_repr<Int, Real>(fil, dualize, oineus::ColumnRepr::BitTree);
        ref.compute_full_u_rows<Real>(dim, value_at, /*n_threads=*/1);

        for (auto cr : {oineus::ColumnRepr::Set, oineus::ColumnRepr::Full,
                        oineus::ColumnRepr::BitTree}) {
            auto d = reduce_col_repr<Int, Real>(fil, dualize, cr);
            d.compute_full_u_rows<Real>(dim, value_at, /*n_threads=*/4);
            REQUIRE(d.u_data_t.size() == ref.u_data_t.size());
            for (size_t r = 0; r < d.u_data_t.size(); ++r)
                REQUIRE(d.u_data_t[r] == ref.u_data_t[r]);
        }

        // Heap has no efficient top(), so the row form must reject it loudly
        // rather than silently fall back or hang.
        auto d_heap = reduce_col_repr<Int, Real>(fil, dualize, oineus::ColumnRepr::Heap);
        REQUIRE_THROWS_AS(
            d_heap.compute_full_u_rows<Real>(dim, value_at, /*n_threads=*/1),
            std::runtime_error);
    }
}


// The parallel ELZ restore (restore_elz_column_parallel_repr) is templated on
// the col_repr-chosen working column; force each of the four instantiations
// to actually RUN under a parallel reduce, on both sides. Unlike the row-form
// U solve, the restore only needs low()/add()/is_zero()/to_vector(), all of
// which HeapColumn provides, so Heap must work here (no throw).
TEST_CASE("parallel restore_elz runs with every col_repr (Set/Heap/Full/BitTree)")
{
    using Int = long;
    using Real = double;
    auto fil = make_test_filtration<Int, Real>(20, 20, /*seed=*/42);

    for (bool dualize : {false, true}) {
        // serial reference: plain ELZ reduction, no clearing
        oineus::VRUDecomposition<Int> ref(fil, dualize);
        oineus::ReductionParams ref_params;
        ref_params.compute_v = true;
        ref_params.use_clearing = false;
        ref_params.n_threads = 1;
        ref.reduce(ref_params);
        auto ref_dgms = ref.diagram(fil, /*include_inf_points=*/true);

        for (auto cr : {oineus::ColumnRepr::Set, oineus::ColumnRepr::Heap,
                        oineus::ColumnRepr::Full, oineus::ColumnRepr::BitTree}) {
            oineus::VRUDecomposition<Int> decmp(fil, dualize);
            oineus::ReductionParams params;
            params.compute_v = true;
            params.use_clearing = true;
            params.n_threads = 4;
            params.advanced.col_repr = cr;
            params.advanced.dims_to_restore_elz = {0, 1};
            decmp.reduce(params);

            // the parallel restore actually ran (it fills the per-worker times)
            REQUIRE(not decmp.dbg_restore_thread_times_.empty());

            // R = DV, low-uniqueness, V upper-triangular after the typed restore
            REQUIRE(decmp.sanity_check());

            // pairing unchanged by the restore: diagrams match the serial reference
            auto dgms = decmp.diagram(fil, /*include_inf_points=*/true);
            REQUIRE(dgms.n_dims() == ref_dgms.n_dims());
            for (dim_type d = 0; d < ref_dgms.n_dims(); ++d) {
                auto got = dgms.get_diagram_in_dimension(d);
                auto expected = ref_dgms.get_diagram_in_dimension(d);
                std::sort(got.begin(), got.end());
                std::sort(expected.begin(), expected.end());
                REQUIRE(got == expected);
            }
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
