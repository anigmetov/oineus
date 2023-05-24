#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <oineus/oineus.h>
#include <opts/opts.h>

using namespace oineus;

template<class Int, class Real, size_t D>
std::tuple<std::vector<Real>, typename oineus::Grid<Int,Real, D>::GridPoint> read_function(std::string fname)
{
    using Grid = oineus::Grid<Int, Real, D>;
    using GridPoint = typename Grid::GridPoint;

    std::ifstream f(fname);

    if (not f.good())
        throw std::runtime_error("Cannot open file " + fname);

    GridPoint dims;

    for(dim_type d = 0; d < D; ++d)
        f >> dims[d];

    size_t n_entries = std::accumulate(dims.cbegin(), dims.cend(), size_t(1), std::multiplies<size_t>());

    std::vector<Real> func;
    func.reserve(n_entries);

    Real x;
    while(f >> x)
        func.push_back(x);

    if (func.size() != n_entries) {
        std::cerr << "Expected " << n_entries << " numbers after dimension, read " << func.size() << std::endl;
        throw std::runtime_error("Bad file format");
    }

    return {func, dims};
}

void test_ls_2()
{
    using IntGrid = Grid<int, double, 2>;

    using IntGridPoint = IntGrid::GridPoint;

    int n_rows = 2;
    int n_cols = 3;

    IntGridPoint dims {n_rows, n_cols};

    std::vector<double> values = {1, 2, 3, 4, 5, 6};
    double* data = values.data();

    bool wrap = false;
    bool negate = false;

    IntGrid grid {dims, wrap, data};

    auto fil = grid.freudenthal_filtration(2, negate);
    std::cout << fil << "\n";
}

void test_ls_3()
{
    using IntGrid = Grid<int, double, 3>;

    using IntGridPoint = IntGrid::GridPoint;

    int n_rows = 4;
    int n_cols = 4;
    int n_zs = 4;
    int size = n_rows * n_cols * n_zs;

    IntGridPoint dims {n_rows, n_cols, n_zs};

    int seed = 1;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<double> values;
    values.reserve(size);
    for(int i = 0; i < size; ++i) {
        values.push_back(dis(gen));
    }

    double* data = values.data();

    bool wrap = true;
    bool negate = false;

    IntGrid grid {dims, wrap, data};

    auto fil = grid.freudenthal_filtration(3, negate);

    VRUDecomposition<int> m_D {fil, false};

    std::cerr << "boundary ok" << std::endl;

    Params params;
    params.n_threads = 4;

    m_D.reduce(params);

    std::cerr << "reduce ok" << std::endl;

    auto dgm = m_D.diagram(fil, true);

    std::cerr << "diagram ok" << std::endl;
}

int main(int argc, char** argv)
{

//    test_ls_3();
//    return 0;
#ifdef OINEUS_USE_SPDLOG
    spdlog::set_level(spdlog::level::info);
#endif

    using opts::Option;
    using opts::PosOption;

    using Int = int;
    using Real = double;

    opts::Options ops;

    std::string fname_in, fname_dgm;
    unsigned int top_d = 2;

    bool help;
    bool bdry_matrix_only {false};
    bool wrap {false};
    bool negate {false};

    Params params;
    ops
            >> Option('d', "dim", top_d, "top dimension")
            >> Option('c', "chunk-size", params.chunk_size, "chunk_size")
            >> Option('t', "threads", params.n_threads, "number of threads")
            >> Option('s', "sort", params.sort_dgms, "sort diagrams")
            >> Option("clear", params.clearing_opt, "clearing optimization")
            >> Option("acq-rel", params.acq_rel, "use acquire-release memory orders")
            >> Option('w', "wrap", wrap, "wrap (periodic boundary conditions)")
            >> Option('n', "negate", negate, "negate function")
            >> Option('m', "matrix-only", bdry_matrix_only, "read boundary matrix w/o filtration")
            >> Option('h', "help", help, "show help message");

    if (!ops.parse(argc, argv) || help || !(ops >> PosOption(fname_in))) {
        std::cout << "Usage: " << argv[0] << " [options] INFILE\n\n";
        std::cout << ops << std::endl;
        return 1;
    }

    info("Reading file {}", fname_in);

    using Grid = oineus::Grid<Int, Real, 3>;

    auto [func, dims] = read_function<Int, Real, 3>(fname_in);
    Grid grid {dims, wrap, func.data()};

    auto fil = grid.freudenthal_filtration(top_d, negate, params.n_threads);
    VRUDecomposition<Int> decmp {fil, false };

    info("Matrix read");

    fname_dgm = fname_in + "_t_" + std::to_string(params.n_threads) + "_c_" + std::to_string(params.chunk_size);

    params.print_time = true;

    decmp.reduce(params);

    if (params.print_time)
        std::cerr << fname_in << ";" << params.n_threads << ";" << params.clearing_opt << ";" << params.chunk_size << ";" << params.elapsed << std::endl;

    auto dgm = decmp.diagram(fil, true);

    if (params.sort_dgms)
        dgm.sort();

    dgm.save_as_txt(fname_dgm);

    info("Diagrams saved");

    return 0;
}
