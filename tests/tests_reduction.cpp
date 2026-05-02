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
    return oineus::Grid<Int, Real, D>(dims, wrap, pdata,
            oineus::Grid<Int, Real, D>::DataLocation::VERTEX);
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
    params.compute_v = true;
    if (n_threads == 1)
        params.compute_u = true;

    decmp.reduce(params);

    auto dgms = decmp.diagram(fil, true);

    return std::make_pair(dgms, decmp);
}

TEST_CASE("Basic reduction")
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

    std::cerr << "Basic reduction: " << decmp << std::endl;

    REQUIRE(decmp.sanity_check());
}


TEST_CASE("Simple reduction parallel")
{
    using Real = double;
    using Int = int;

    std::vector<Real> xs;

    std::ifstream f {"./a_6.txt"};

    REQUIRE(f.good());

    Real x;
    while(f >> x)
        xs.push_back(x);

    REQUIRE(xs.size() == 216);

    using Grid = oineus::Grid<int, Real, 3>;
    using GridPoint = typename Grid::GridPoint;

    bool wrap = false;
    bool negate = true;
    int n_threads = 4;
    dim_type top_d = 2;

    GridPoint dims {6, 6, 6};

    Grid grid = get_grid<Int, Real, 3>(xs.data(), dims, wrap);
    auto dv = compute_diagrams_and_v_ls_freudenthal<Int, Real, 3>(grid, negate, wrap, top_d, n_threads);
    auto dgms = dv.first;
    auto decmp = dv.second;
    REQUIRE(dgms.n_dims() == top_d + 2);
    REQUIRE(decmp.sanity_check());
}


TEST_CASE("Vietoris--Rips")
{
    using Real = double;
    using Int = int;
    using oineus::dim_type;
    using Point = oineus::Point<Real, 2>;

    std::vector<Point> points;
    int n_points = 6;
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<Real> dis(0, 1);
    for(int i = 0; i < n_points; ++i) {
        Point p;
        for(int coord = 0; coord < 2; ++coord)
            p[coord]  = dis(gen);
        points.push_back(p);
//        points.emplace_back({dis(gen), dis(gen)});
    }

    dim_type max_dim = 3;
    Real max_radius = 0.4;

    auto fil_1 = oineus::get_vr_filtration<Int, Real, 2>(points, max_dim, max_radius);
    auto fil_2 = oineus::get_vr_filtration_naive<Int, Real, 2>(points, max_dim, max_radius);

    for(size_t i = 0; i < fil_1.size(); ++i) {
        REQUIRE(fil_1.get_cell(i).get_uid() == fil_2.get_cell(i).get_uid());
    }
}


TEST_CASE("Vietoris--Rips in-order matches Bron-Kerbosch")
{
    using Real = double;
    using Int = int;
    using oineus::dim_type;
    using Point = oineus::Point<Real, 2>;

    std::vector<Point> points;
    int n_points = 8;
    std::mt19937_64 gen(2);
    std::uniform_real_distribution<Real> dis(0, 1);
    for(int i = 0; i < n_points; ++i) {
        Point p;
        for(int coord = 0; coord < 2; ++coord)
            p[coord] = dis(gen);
        points.push_back(p);
    }

    dim_type max_dim = 3;
    Real max_radius = 0.7;

    auto fil_bk  = oineus::get_vr_filtration<Int, Real, 2>(points, max_dim, max_radius);
    auto fil_vre = oineus::get_vr_filtration_inorder<Int, Real, 2>(points, max_dim, max_radius);

    REQUIRE(fil_bk.size() == fil_vre.size());

    // Compare cell sets (uid + value); ordering may differ for ties but the
    // multiset must match.
    using Pair = std::pair<unsigned __int128, Real>;
    std::vector<Pair> bk_cells, vre_cells;
    bk_cells.reserve(fil_bk.size());
    vre_cells.reserve(fil_vre.size());
    for(size_t i = 0; i < fil_bk.size(); ++i)
        bk_cells.emplace_back(fil_bk.get_cell(i).get_uid(), fil_bk.get_cell(i).get_value());
    for(size_t i = 0; i < fil_vre.size(); ++i)
        vre_cells.emplace_back(fil_vre.get_cell(i).get_uid(), fil_vre.get_cell(i).get_value());
    auto cmp = [](const Pair& a, const Pair& b) {
        return a.first < b.first;
    };
    std::sort(bk_cells.begin(), bk_cells.end(), cmp);
    std::sort(vre_cells.begin(), vre_cells.end(), cmp);
    for(size_t i = 0; i < bk_cells.size(); ++i) {
        REQUIRE(bk_cells[i].first == vre_cells[i].first);
        REQUIRE(std::abs(bk_cells[i].second - vre_cells[i].second) < Real(1e-12));
    }
}
