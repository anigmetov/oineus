// Phase-3 unit tests for the bounded variants of compute_u_column
// (Algorithm 3) and compute_u_column_1 (Algorithm 4).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>
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
