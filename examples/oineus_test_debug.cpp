#include <iostream>
#include <fstream>
#include <oineus/oineus.h>

using Real = double;
using Int = int;
using namespace oineus;

template<class Int, class Real, size_t D>
typename oineus::Grid<Int, Real, D>
get_grid(Real* pdata, typename oineus::Grid<Int, Real, D>::GridPoint dims, bool wrap)
{
    return oineus::Grid<Int, Real, D>(dims, wrap, pdata);
}

template<class Int, class Real, size_t D>
decltype(auto) compute_diagrams_and_v_ls_freudenthal(const typename oineus::Grid<Int, Real, D>& grid, bool negate, bool wrap, dim_type top_d, int n_threads)
{
    auto fil = grid.freudenthal_filtration(top_d + 1, negate, n_threads).first;
    auto decmp = oineus::VRUDecomposition<Int>(fil, false);

    oineus::Params params;

    params.sort_dgms = false;
    params.clearing_opt = false;
    params.n_threads = n_threads;
    params.compute_u = true;

    decmp.reduce(params);

    auto dgms = decmp.diagram(fil, true);

    return std::make_pair(dgms, decmp);
}
int main()
{
    std::vector<Real> xs;

    std::ifstream f {"/home/narn/code/oineus/tests/a_6.txt"};

    Real x;
    while(f >> x)
        xs.push_back(x);

    assert(xs.size() == 216);

    using Grid = oineus::Grid<int, Real, 3>;
    using GridPoint = typename Grid::GridPoint;

    bool wrap = false;
    bool negate = true;
    int n_threads = 1;
    dim_type top_d = 2;

    GridPoint dims {6, 6, 6};

    Grid grid = get_grid<Int, Real, 3>(xs.data(), dims, wrap);
    auto dv = compute_diagrams_and_v_ls_freudenthal<Int, Real, 3>(grid, negate, wrap, top_d, n_threads);
    auto dgms = dv.first;
    auto decmp = dv.second;
    assert(dgms.n_dims() == top_d + 1);
    assert(decmp.sanity_check());
    return 0;
}


