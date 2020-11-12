#include <iostream>
#include <atomic>
#include <vector>
#include <string>
#include <thread>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>

#include <stdlib.h>

#include <oineus/includes.h>

#include "opts/opts.h"

using namespace oineus;


void test_ls_2()
{
    using IntGrid = Grid<int, double, 2>;

    using IntGridPoint = IntGrid::GridPoint;

    int n_rows = 2;
    int n_cols = 3;

    IntGridPoint dims {n_rows, n_cols};

    std::vector<double> values = { 1, 2, 3, 4, 5, 6 };
    double* data = values.data();

    bool wrap = false;
    bool negate = false;

    Grid<int, double, 2> grid { dims, wrap, data };

    auto ss = grid.freudenthal_simplices(2, negate);

    for(const auto& s : ss)
        std::cout << s << "\n";

    auto fil = grid.freudenthal_filtration(2, negate);
    std::cout << fil << "\n";
}


void test_ls_3()
{
    using IntGrid = Grid<int, double, 3>;

    using IntGridPoint = IntGrid::GridPoint;

    int n_rows = 3;
    int n_cols = 3;
    int n_zs = 3;

    IntGridPoint dims {n_rows, n_cols, n_zs};

    std::vector<double> values;
    for(int i =0; i < n_rows * n_cols * n_zs; ++i)
        values.push_back(i);

    double* data = values.data();

    bool wrap = true;
    bool negate = false;

    Grid<int, double, 3> grid { dims, wrap, data };

    auto ss = grid.freudenthal_simplices(2, negate);

    std::cout << "2-skeleton:\n";
    for(const auto& s : ss)
        std::cout << s << "\n";

    std::cout << "-----------------------------------------------\n";

    std::cout << "3-skeleton:\n";

    ss = grid.freudenthal_simplices(3, negate);

    for(const auto& s : ss)
        std::cout << s << "\n";


}


int main(int argc, char** argv)
{

    test_ls_2();
    return 0;
#ifdef ARACHNE_USE_SPDLOG
    spdlog::set_level(spdlog::level::info);
#endif

    using opts::Option;
    using opts::PosOption;

    using Int = int;
    using Real = double;

    opts::Options ops;

    std::string fname_in, fname_dgm;
    unsigned int max_dim = 1;
    int n_threads { 1 };

    bool help;
    bool bdry_matrix_only { false };

    Params params;
    ops
            >> Option('d', "dim", max_dim, "top dimension")
            >> Option('c', "chunk-size", params.chunk_size, "chunk_size")
            >> Option('t', "threads", n_threads, "number of threads")
            >> Option('s', "sort", params.sort_dgms, "sort diagrams")
            >> Option(     "clear", params.clearing_opt, "clearing optimization")
            >> Option(     "acq-rel", params.acq_rel, "use acquire-release memory orders")
            >> Option('m', "matrix-only", bdry_matrix_only, "read boundary matrix w/o filtration")
            >> Option('h', "help", help, "show help message");

    if (!ops.parse(argc, argv) || help || !(ops >> PosOption(fname_in))) {
        std::cout << "Usage: " << argv[0] << " [options] INFILE\n\n";
        std::cout << ops << std::endl;
        return 1;
    }

    Filtration<Int, Real> fil;

    info("Reading file {}", fname_in);

    //read_filtration(fname_in, fil, params.clearing_opt);
    SparseMatrix<Int> r = fil.boundary_matrix_full();

    info("Matrix read");

    fname_dgm = fname_in + "_t_" + std::to_string(n_threads) + "_c_" + std::to_string(params.chunk_size);

    r.reduce_parallel(n_threads, params);

    if (params.print_time) {
        std::cerr << fname_in << ";" <<  n_threads << ";" << params.clearing_opt << ";" << params.chunk_size << ";" << params.elapsed << std::endl;
    }

    auto dgm = r.diagram(fil);

    if (params.sort_dgms)
        dgm.sort();

    dgm.save_as_txt(fname_dgm);

    info("Diagrams saved");

    return 0;
}
