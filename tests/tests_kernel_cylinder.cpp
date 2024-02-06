#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <fstream>
#include <random>
#include <oineus/oineus.h>

using dim_type = size_t;

template<class Int, class Real, size_t D>
typename oineus::Grid<Int, Real, D>
get_grid(Real* pdata, typename oineus::Grid<Int, Real, D>::GridPoint dims, bool wrap)
{
    return oineus::Grid<Int, Real, D>(dims, wrap, pdata);
}

template<class Int, class Real, size_t D>
typename oineus::VRUDecomposition<Int>::MatrixData
get_boundary_matrix(const typename oineus::Grid<Int, Real, D>& grid, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = grid.freudenthal_filtration(top_d, negate, n_threads);
    auto bm = fil.boundary_matrix_full();
    return bm.data;
}

template<class Int, class Real, size_t D>
decltype(auto) compute_diagrams_and_v_ls_freudenthal(const typename oineus::Grid<Int, Real, D>& grid, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = grid.freudenthal_filtration(top_d + 1, negate, n_threads);
    auto decmp = oineus::VRUDecomposition<Int>(fil, false);

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = false;
    params.n_threads = n_threads;
    if (n_threads == 1)
        params.compute_u = true;

    decmp.reduce(params);

    auto dgms = decmp.diagram(fil, true);

    return std::make_pair(dgms, decmp);
}

TEST_CASE("Kernel")
{
    using Int = int;

    std::vector<std::vector<Int>> d_cols;
    d_cols.emplace_back(std::vector<Int>({0, 1}));
    d_cols.emplace_back(std::vector<Int>({1}));

    oineus::VRUDecomposition<Int> decmp(d_cols);

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = false;
    params.compute_v = true;
    params.compute_u = true;
    params.n_threads = 1;

    decmp.reduce(params);

    REQUIRE(decmp.sanity_check());
}




