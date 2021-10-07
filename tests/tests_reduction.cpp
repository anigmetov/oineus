#include <catch2/catch.hpp>

#include <iostream>
#include <fstream>
#include <oineus/oineus.h>

using dim_type = size_t;

template<class Int, class Real, size_t D>
typename oineus::Grid<Int, Real, D>
get_grid(Real* pdata, typename oineus::Grid<Int, Real, D>::GridPoint dims, bool wrap)
{
    return oineus::Grid<Int, Real, D>(dims, wrap, pdata);
}


template<class Int, class Real, size_t D>
typename oineus::SparseMatrix<Int>::MatrixData
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
    auto d_matrix = fil.boundary_matrix_full();

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = true;
    params.n_threads = n_threads;

    d_matrix.reduce_parallel(params);

    auto dgms = d_matrix.diagram(fil);

    return std::make_pair(dgms, d_matrix.v_data);
}

//
//template<class Int, class Real, size_t D>
//PyOineusDiagrams<Real>
//compute_diagrams_ls_freudenthal(py::array_t<Real> data, bool negate, bool wrap, dim_type top_d, int n_threads)
//{
//    std::cerr << "enter" << std::endl;
//    auto fil = get_filtration<Int, Real, D>(data, negate, wrap, top_d, n_threads);
//    auto d_matrix = fil.boundary_matrix_full();
//    std::cerr << "matrix ok" << std::endl;
//
//    oineus::Params params;
//
//    params.sort_dgms = false;
//    params.clearing_opt = true;
//    params.n_threads = n_threads;
//
//    d_matrix.reduce_parallel(params);
//
//    std::cerr << "reduction ok" << std::endl;
//
//    return PyOineusDiagrams<Real>(d_matrix.diagram(fil));
//}


TEST_CASE("Simple reduction")
{
    using Real = double;
    using Int = int;

    std::vector<Real> xs;

    std::ifstream f { "/home/narn/code/oineus/tests/a_6.txt" };

    Real x;
    while(f >> x)
        xs.push_back(x);

    REQUIRE( xs.size() == 216 );

    using Grid = oineus::Grid<int, Real, 3>;
    using GridPoint = typename Grid::GridPoint;

    bool wrap = false;
    bool negate = true;
    int n_threads = 1;
    dim_type top_d = 2;

    GridPoint dims { 6, 6, 6};

    Grid grid = get_grid<Int, Real, 3>(xs.data(), dims, wrap);
    auto dv = compute_diagrams_and_v_ls_freudenthal<Int, Real, 3>(grid, negate, wrap, top_d, n_threads);
    auto dgms = dv.first;
    REQUIRE( dgms.n_dims() == top_d + 1 );
}