#pragma once

#include <iostream>
#include <atomic>
#include <vector>
#include <string>
#include <thread>
#include <pthread.h>
#include <cassert>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <unordered_set>
#include <set>
#include <cstdlib>
#include <stdexcept>

#include "icecream/icecream.hpp"
#include "common_defs.h"
#include "timer.h"
#include "diagram.h"
#include "mem_reclamation.h"
#include "sparse_matrix.h"

namespace oineus {

template<typename Int_, typename Real_, typename L_>
class Filtration;


// return index of the lowest non-zero in column i of r, -1 if empty
template<typename IdxType>
IdxType low(const SparseColumn<IdxType>* c)
{
    return c->empty() ? -1 : c->back();
}

template<typename IdxType>
IdxType low(const SparseColumn<IdxType>& c)
{
    return c.empty() ? -1 : c.back();
}

template<typename IdxType>
bool is_zero(const SparseColumn<IdxType>& c)
{
    return c.empty();
}

template<typename IdxType>
bool is_zero(const SparseColumn<IdxType>* c)
{
    return c->empty();
}

template<typename IdxType>
void add_column(const SparseColumn<IdxType>& col_a, const SparseColumn<IdxType>& col_b, SparseColumn<IdxType>& sum)
{
    sum.clear();
    std::set_symmetric_difference(col_a.begin(), col_a.end(), col_b.begin(), col_b.end(), std::back_inserter(sum));
}

template<typename IdxType>
void add_column_1(const SparseColumn<IdxType>& col_a, const SparseColumn<IdxType>& col_b, SparseColumn<IdxType>& sum)
{
    // add_column cannot work as +=
    assert(col_a.data() != sum.data() and col_b.data() != sum.data());

    auto a_iter = col_a.cbegin();
    auto b_iter = col_b.cbegin();

    sum.clear();

    while(true) {
        if (a_iter == col_a.cend() && b_iter == col_b.cend()) {
            break;
        } else if (a_iter == col_a.cend() && b_iter != col_b.cend()) {
            sum.push_back(*b_iter++);
        } else if (a_iter != col_a.cend() && b_iter == col_b.cend()) {
            sum.push_back(*a_iter++);
        } else if (*a_iter < *b_iter) {
            sum.push_back(*a_iter++);
        } else if (*b_iter < *a_iter) {
            sum.push_back(*b_iter++);
        } else {
            assert(*a_iter == *b_iter);
            ++a_iter;
            ++b_iter;
        }
    }
}

template<typename IdxType_>
struct RVColumns {
    using IdxType = IdxType_;
    using Column = SparseColumn<IdxType>;

    Column r_column;
    Column v_column;

    RVColumns() = default;
    RVColumns(const RVColumns& other) = default;
    RVColumns(RVColumns&& other) noexcept = default;
    RVColumns(Column&& _r, Column&& _v)
            :r_column(_r), v_column(_v) { }

    RVColumns(const Column& _r, const Column& _v)
            :r_column(_r), v_column(_v) { }

    [[nodiscard]] bool is_zero() const
    {
        return oineus::is_zero(&r_column);
    }

    IdxType low() const
    {
        return oineus::low(&r_column);
    }

    void swap(RVColumns& other) noexcept
    {
        r_column.swap(other.r_column);
        v_column.swap(other.v_column);
    }

    template<class I>
    friend std::ostream& operator<<(std::ostream& out, const RVColumns<I>& rv);
};

template<class I>
std::ostream& operator<<(std::ostream& out, const RVColumns<I>& rv)
{
    out << "R column: [";
    for(auto x : rv.r_column)
        out << x << ", ";

    out << "], V column: [";
    for(auto x : rv.v_column)
        out << x << ", ";

    out << "]";

    return out;
}


template<class I>
std::ostream& operator<<(std::ostream& out, const std::atomic<RVColumns<I>*>& rv)
{
    out << *rv;
    return out;
}

template<class I>
std::ostream& operator<<(std::ostream& out, const std::vector<std::atomic<RVColumns<I>*>>& rvs)
{
    out << "RV matrix:\n";
    for(const auto& rv : rvs)
        out << rv << "\n";
    out << "\n";
    return out;
}

template<typename IdxType>
void add_rv_column(const RVColumns<IdxType>* col_a, const RVColumns<IdxType>* col_b, RVColumns<IdxType>* sum)
{
    add_column(col_a->r_column, col_b->r_column, sum->r_column);
    add_column(col_a->v_column, col_b->v_column, sum->v_column);
}

template<class RVMatrices, class AtomicIdxVector, class Int, class MemoryReclaimC>
void parallel_reduction(RVMatrices& rv, std::vector<SparseColumn<Int>>& u_rows, AtomicIdxVector& pivots, std::atomic<Int>& next_free_chunk,
        const Params params, int thread_idx, MemoryReclaimC* mm, ThreadStats& stats, bool go_down)
{
    using RVColumnC = RVColumns<Int>;
    using PRVColumn = RVColumnC*;

    std::memory_order acq = params.acq_rel ? std::memory_order_acquire : std::memory_order_seq_cst;
    std::memory_order rel = params.acq_rel ? std::memory_order_release : std::memory_order_seq_cst;
    std::memory_order relax = params.acq_rel ? std::memory_order_relaxed : std::memory_order_seq_cst;

    debug("thread {} started, mm = {}", thread_idx, (void*) (mm));
    std::unique_ptr<RVColumnC> reduced_r_v_column(new RVColumnC);
    std::unique_ptr<RVColumnC> reduced_r_v_column_final(new RVColumnC);

    const int n_cols = rv.size();

    do {
        int my_chunk;

        if (go_down) {
            my_chunk = next_free_chunk--;
        } else {
            my_chunk = next_free_chunk++;
        }

        int chunk_begin = std::max(0, my_chunk * params.chunk_size);
        int chunk_end = std::min(n_cols, (my_chunk + 1) * params.chunk_size);

        if (chunk_begin >= n_cols || chunk_end <= 0) {
            debug("Thread {} finished", thread_idx);
            return;
        }

        debug("thread {}, processing chunk {}, from {} to {}, n_cols = {}", thread_idx, my_chunk, chunk_begin, chunk_end, n_cols);

        int current_column_idx = chunk_begin;
        int next_column = current_column_idx + 1;

        while(current_column_idx < chunk_end) {

            debug("thread {}, column = {}", thread_idx, current_column_idx);

            PRVColumn current_r_v_column = rv[current_column_idx].load(acq);
            PRVColumn original_r_v_column = current_r_v_column;

            bool update_column = false;

            int pivot_idx;

            if (params.clearing_opt) {
                if (!current_r_v_column->is_zero()) {
                    int c_pivot_idx = pivots[current_column_idx].load(acq);
                    if (c_pivot_idx >= 0) {
                        // unset pivot from current_column_idx, if necessary
                        int c_current_low = current_r_v_column->low();
                        int c_current_column_idx = current_column_idx;
                        pivots[c_current_low].compare_exchange_weak(c_current_column_idx, -1, rel, relax);

                        // zero current column
                        current_r_v_column = new RVColumnC();
                        rv[current_column_idx].store(current_r_v_column, rel);
                        mm->retire(original_r_v_column);
                        original_r_v_column = current_r_v_column;

                        stats.n_cleared++;
                    }
                }
            }

            while(!current_r_v_column->is_zero()) {

                int current_low = current_r_v_column->low();
                PRVColumn pivot_r_v_column = nullptr;

                debug("thread {}, column = {}, low = {}", thread_idx, current_column_idx, current_low);

                do {
                    pivot_idx = pivots[current_low].load(acq);
                    if (pivot_idx >= 0) {
                        pivot_r_v_column = rv[pivot_idx].load(acq);
                    }
                }
                while(pivot_idx >= 0 && pivot_r_v_column->low() != current_low);

                if (pivot_idx == -1) {
                    if (pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                        break;
                    }
                } else if (pivot_idx < current_column_idx) {
                    // pivot to the left: kill lowest one in current column
                    add_rv_column(current_r_v_column, pivot_r_v_column, reduced_r_v_column.get());
                    SparseColumn<Int> new_u;
                    add_column(u_rows[pivot_idx], u_rows[current_column_idx], new_u);
                    u_rows[pivot_idx] = std::move(new_u);

                    update_column = true;

                    reduced_r_v_column_final->swap(*reduced_r_v_column);
                    current_r_v_column = reduced_r_v_column_final.get();
                } else if (pivot_idx > current_column_idx) {

                    stats.n_right_pivots++;

                    // pivot to the right: switch to reducing r[pivot_idx]
                    if (update_column) {
                        // create copy of reduced column and write in into matrix
                        auto new_r_v_column = new RVColumnC(*reduced_r_v_column_final);
                        rv[current_column_idx].store(new_r_v_column, rel);
                        // original column can be deleted
                        mm->retire(original_r_v_column);
                        original_r_v_column = new_r_v_column;
                        update_column = false;
                    }

                    // set current column as new pivot, start reducing column r[pivot_idx]
                    if (pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                        current_column_idx = pivot_idx;
                        current_r_v_column = rv[current_column_idx].load(acq);
                        original_r_v_column = current_r_v_column;
                    }
                }
            } // reduction loop

            if (update_column) {
                // write copy of reduced column to matrix
                // TODO: why not use reduced_r_column_final directly?
                current_r_v_column = new RVColumnC(*reduced_r_v_column_final);
                rv[current_column_idx].store(current_r_v_column, rel);
                mm->retire(original_r_v_column);
            }

            current_column_idx = next_column;
            next_column = current_column_idx + 1;

        } //loop over columns

        mm->quiescent();
    }
    while(true); // loop over chunks
}

template<typename Int_>
struct VRUDecomposition {
    // types
    using Int = Int_;
    using IntSparseColumn = SparseColumn<Int>;
    using MatrixData = std::vector<IntSparseColumn>;
    using AtomicIdxVector = std::vector<std::atomic<Int>>;

    // data
    MatrixData d_data;
    MatrixData r_data;
    MatrixData v_data;
    MatrixData u_data_t;
    bool is_reduced {false};
    const bool dualize_ {false};

    std::vector<size_t> dim_first;
    std::vector<size_t> dim_last;

    // methods

    template<class R, class L>
    VRUDecomposition(const Filtration<Int, R, L>& fil, bool _dualize) :
            d_data (!_dualize ? fil.boundary_matrix_full() : antitranspose(fil.boundary_matrix_full(), fil.size())),
            r_data (d_data),
            dualize_(_dualize),
            dim_first(fil.dim_first()),
            dim_last(fil.dim_last())
    {
        if (dualize_) {
            std::reverse(dim_first.begin(), dim_first.end());
            std::reverse(dim_last.begin(), dim_last.end());
            std::vector<size_t> new_dim_first, new_dim_last;
            for(size_t i = 0; i < dim_first.size(); ++i) {
                size_t cnt = dim_last[i] - dim_first[i];
                if (i == 0) {
                    new_dim_first.push_back(0);
                    new_dim_last.push_back(cnt);
                } else {
                    new_dim_first.push_back(new_dim_last.back() + 1);
                    new_dim_last.push_back(new_dim_first.back() + cnt);
                }
            }
            dim_first = new_dim_first;
            dim_last = new_dim_last;
        }
    }

    VRUDecomposition(const MatrixData& d) :
            d_data (d),
            r_data (d),
            dualize_(false),
            dim_first(std::vector<size_t>({0})),
            dim_last(std::vector<size_t>({d.size()-1}))
    {
    }

    [[nodiscard]] size_t size() const { return r_data.size(); }

    bool dualize() const { return dualize_; }

//    void append(VRUDecomposition&& other);

    void reduce(Params& params);

    void reduce_serial(Params& params);

    void reduce_parallel_r_only(Params& params);
    void reduce_parallel_rv(Params& params);
    void reduce_parallel_rvu(Params& params);

    template<typename Real, typename L>
    Diagrams<Real> diagram(const Filtration<Int, Real, L>& fil, bool include_inf_points) const;

    template<typename Real, typename L>
    Diagrams<size_t> index_diagram(const Filtration<Int, Real, L>& fil, bool include_inf_points, bool include_zero_persistence_points) const;

    template<typename Int>
    friend std::ostream& operator<<(std::ostream& out, const VRUDecomposition<Int>& m);

    bool sanity_check();
};

template<class Int>
bool are_matrix_columns_sorted(const std::vector<std::vector<Int>>& matrix_cols)
{
    for(auto&& col : matrix_cols) {
        if (not std::is_sorted(col.begin(), col.end())) {
            for(auto e : col)
                std::cerr << "NOT SORTED: " << e << std::endl;
            return false;
        }
    }
    return true;
}

template<class Int>
bool is_matrix_reduced(const std::vector<std::vector<Int>>& matrix_cols)
{
    std::unordered_set<Int> lowest_ones;

    for(auto&& col : matrix_cols) {
        if (is_zero(col))
            continue;
        Int lo = low(col);
        if (lowest_ones.count(lo))
            return false;
        else
            lowest_ones.insert(lo);
    }
    return true;
}


template<class Int>
bool do_rows_and_columns_match(const std::vector<std::vector<Int>>& matrix_cols, const std::vector<std::vector<Int>>& matrix_rows)
{
    if (matrix_cols.empty())
        return matrix_rows.empty();

    std::set<std::pair<size_t, size_t>> e_rows, e_cols;

    for(size_t col_idx = 0; col_idx < matrix_cols.size(); ++col_idx)
        for(auto row_idx : matrix_cols[col_idx])
            e_cols.emplace(static_cast<size_t>(row_idx), col_idx);

     for(size_t row_idx = 0; row_idx < matrix_rows.size(); ++row_idx)
        for(auto col_idx : matrix_rows[row_idx])
            e_rows.emplace(row_idx, static_cast<size_t>(col_idx));

    return e_cols == e_rows;
}


template<class Int>
bool VRUDecomposition<Int>::sanity_check()
{
    bool verbose = true;
    // R is reduced
    if (not is_matrix_reduced(r_data)) {
        std::cerr << "sanity_check: R not reduced!" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "R reduced" << std::endl;

    // R is sorted
    if (not are_matrix_columns_sorted(r_data)) {
        std::cerr << "sanity_check: R not sorted!" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "R sorted" << std::endl;

    // V is sorted
    if (not are_matrix_columns_sorted(v_data)) {
        std::cerr << "sanity_check: V not sorted!" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "V sorted" << std::endl;

    // U is sorted
    if (not are_matrix_columns_sorted(u_data_t)) {
        std::cerr << "sanity_check: U not sorted!" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "U sorted" << std::endl;

    size_t n_simplices = r_data.size();

    // all matrices are square
    SparseMatrix<Int> dd (d_data, n_simplices, true);
    SparseMatrix<Int> rr (r_data, n_simplices, true);
    SparseMatrix<Int> uu (u_data_t,  n_simplices, false);
    SparseMatrix<Int> vv (v_data, n_simplices, true);

    SparseMatrix<Int> ii = eye<Int>(n_simplices);

    SparseMatrix<Int> dv = mat_multiply_2(dd, vv);
    if (verbose) std::cerr << "DV computed" << std::endl;

    SparseMatrix<Int> uv = mat_multiply_2(uu, vv);
    if (verbose) std::cerr << "UV computed" << std::endl;

    if (not vv.is_upper_triangular()) {
        std::cerr << "V not upper-triangular" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "V upper-triangular" << std::endl;

    if (not uu.is_upper_triangular()) {
        std::cerr << "U not upper-triangular" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "U upper-triangular" << std::endl;

   if (uv != ii) {
        std::cerr << "UV != I" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "UV = I" << std::endl;

    if (dv != rr) {
        std::cerr << "R != DV" << std::endl;
        return false;
    }
    if (verbose) std::cerr << "R = DV" << std::endl;

    return true;
}

template<class Int>
void VRUDecomposition<Int>::reduce(Params& params)
{
    if (params.n_threads == 1)
        reduce_serial(params);
    else
        reduce_parallel_rvu(params);
//    else if (params.compute_v and params.compute_u)
//        reduce_parallel_rvu(params);
//    else if (params.compute_v)
//        reduce_parallel_rv(params);
//    else
//        reduce_parallel_r_only(params);

}

template<class Int>
void VRUDecomposition<Int>::reduce_serial(Params& params)
{
    Timer timer, timer_total;
    int n_cleared = 0;

    Int n_cols = d_data.size();

    if (params.compute_v) {
        v_data = MatrixData(d_data.size());
        for(size_t i = 0; i < d_data.size(); ++i) {
            v_data[i].push_back(static_cast<Int>(i));
        }
    }
    if (params.compute_u) {
        u_data_t = MatrixData(d_data.size());
        for(size_t i = 0; i < d_data.size(); ++i) {
            u_data_t[i].push_back(static_cast<Int>(i));
        }
    }

    std::vector<Int> pivots(d_data.size(), -1);
    assert(pivots.size() == d_data.size() and pivots.size() > 0);

    auto init_time = timer.elapsed_reset();

    // homology: go from top dimension to 0, to make clearing possible
    // cohomology:
    IntSparseColumn new_col;
    for(int dim = dim_first.size() - 1; dim >= 0; --dim) {
        for(Int i = dim_first[dim]; i <= dim_last[dim]; ++i) {
//            if (i == dim_first[dim]) { IC(dim, dim_first[dim], dim_last[dim], dualize_, r_data[i].size()); }

//            IC(dim, i, r_data[i]);

            if (params.clearing_opt and not is_zero(r_data[i])) {
                // simplex i is pivot -> i is positive -> its column is 0
                if (pivots[i] >= 0) {
                    assert(pivots[low(r_data[i])] == -1);
                    r_data[i].clear();
                    n_cleared++;
                    continue;
                }
            }

            while(not is_zero(r_data[i])) {

                Int& pivot = pivots[low(r_data[i])];

                if (pivot == -1) {
                    pivot = i;
                    break;
                } else {
                    add_column(r_data[i], r_data[pivot], new_col);
                    r_data[i] = std::move(new_col);

                    if (params.compute_v) {
                        add_column(v_data[i], v_data[pivot], new_col);
                        v_data[i] = std::move(new_col);
                    }

                    if (params.compute_u)
                        u_data_t[pivot].push_back(i);
                }
            } // reduction loop
        } // loop over columns in fixed dimension
    } // loop over dimensions

    auto reduction_time = timer.elapsed_reset();
    params.elapsed = timer_total.elapsed_reset();

    if (params.print_time) std::cerr << "reduce_serial, matrix_size = " << r_data.size() << ", clearing_opt = " << params.clearing_opt << ", n_cleared = " << n_cleared << ", total elapsed: " << params.elapsed << ", reduction_time = " << reduction_time << std::endl;

    is_reduced = true;
}


template<class Int>
void VRUDecomposition<Int>::reduce_parallel_r_only(Params& params)
{
    using namespace std::placeholders;

    size_t n_cols = size();

    using RVColumnC = RVColumns<Int>;
    using APRVColumn = std::atomic<RVColumnC*>;
    using RVMatrix = std::vector<APRVColumn>;
    using MemoryReclaimC = MemoryReclaim<RVColumnC>;

    RVMatrix r_v_matrix(n_cols);

//    IC("Before moving: ", d_data);

    // move data to r_v_matrix
    for(size_t i = 0; i < n_cols; ++i) {
        IntSparseColumn v_column = {static_cast<Int>(i)};
        IntSparseColumn u_row = {static_cast<Int>(i)};
        u_data_t.push_back(u_row);
        r_v_matrix[i] = new RVColumnC(r_data[i], v_column);
    }
    debug("Matrix moved");

//    IC(r_v_matrix, d_data);

    std::atomic<Int> counter;
    counter = 0;

    std::atomic<Int> next_free_chunk;

    AtomicIdxVector pivots(n_cols);
    for(auto& p: pivots) {
        p.store(-1, std::memory_order_relaxed);
    }
    debug("Pivots initialized");

    int n_threads = std::min(params.n_threads, std::max(1, static_cast<int>(n_cols / params.chunk_size)));

    std::vector<std::thread> ts;
    std::vector<std::unique_ptr<MemoryReclaimC>> mms;
    std::vector<ThreadStats> stats;

    mms.reserve(n_threads);
    stats.reserve(n_threads);

    bool go_down = false;

    if (go_down) {
        next_free_chunk = n_cols / params.chunk_size;
    } else {
        next_free_chunk = 0;
    }

    Timer timer;

    for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

        mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
        stats.emplace_back(thread_idx);

        ts.emplace_back(parallel_reduction<RVMatrix, AtomicIdxVector, Int, MemoryReclaimC>,
                std::ref(r_v_matrix), std::ref(u_data_t), std::ref(pivots), std::ref(next_free_chunk),
                params, thread_idx, mms[thread_idx].get(), std::ref(stats[thread_idx]), go_down);

#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_idx, &cpuset);
        int rc = pthread_setaffinity_np(ts[thread_idx].native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
#endif
    }

    info("{} threads created", ts.size());

    for(auto& t: ts) {
        t.join();
    }

    params.elapsed = timer.elapsed_reset();

    if (params.print_time) {
        for(auto& s: stats) { info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots); }
        info("n_threads = {}, chunk = {}, elapsed = {} sec", n_threads, params.chunk_size, params.elapsed);
        std::cerr << "n_threads = " << n_threads << ", elapsed = " << params.elapsed << std::endl;
    }

    // write reduced matrix back, collect V matrix, mark as reduced
    for(size_t i = 0; i < n_cols; ++i) {
        auto p = r_v_matrix[i].load(std::memory_order_relaxed);
        r_data[i] = std::move(p->r_column);
        v_data[i] = std::move(p->v_column);
    }

//    IC("After R writen back: ", d_data);

    is_reduced = true;
}



template<class Int>
void VRUDecomposition<Int>::reduce_parallel_rvu(Params& params)
{
    using namespace std::placeholders;

    size_t n_cols = size();

    v_data = std::vector<IntSparseColumn>(n_cols);
    u_data_t.reserve(n_cols);

    using RVColumnC = RVColumns<Int>;
    using APRVColumn = std::atomic<RVColumnC*>;
    using RVMatrix = std::vector<APRVColumn>;
    using MemoryReclaimC = MemoryReclaim<RVColumnC>;

    RVMatrix r_v_matrix(n_cols);

//    IC("Before moving: ", d_data);

    // move data to r_v_matrix
    for(size_t i = 0; i < n_cols; ++i) {
        IntSparseColumn v_column = {static_cast<Int>(i)};
        IntSparseColumn u_row = {static_cast<Int>(i)};
        u_data_t.push_back(u_row);
        r_v_matrix[i] = new RVColumnC(r_data[i], v_column);
    }
    debug("Matrix moved");

//    IC(r_v_matrix, d_data);

    std::atomic<Int> counter;
    counter = 0;

    std::atomic<Int> next_free_chunk;

    AtomicIdxVector pivots(n_cols);
    for(auto& p: pivots) {
        p.store(-1, std::memory_order_relaxed);
    }
    debug("Pivots initialized");

    int n_threads = std::min(params.n_threads, std::max(1, static_cast<int>(n_cols / params.chunk_size)));

    std::vector<std::thread> ts;
    std::vector<std::unique_ptr<MemoryReclaimC>> mms;
    std::vector<ThreadStats> stats;

    mms.reserve(n_threads);
    stats.reserve(n_threads);

    bool go_down = false;

    if (go_down) {
        next_free_chunk = n_cols / params.chunk_size;
    } else {
        next_free_chunk = 0;
    }

    Timer timer;

    for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

        mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
        stats.emplace_back(thread_idx);

        ts.emplace_back(parallel_reduction<RVMatrix, AtomicIdxVector, Int, MemoryReclaimC>,
                std::ref(r_v_matrix), std::ref(u_data_t), std::ref(pivots), std::ref(next_free_chunk),
                params, thread_idx, mms[thread_idx].get(), std::ref(stats[thread_idx]), go_down);

#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_idx, &cpuset);
        int rc = pthread_setaffinity_np(ts[thread_idx].native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
#endif
    }

    info("{} threads created", ts.size());

    for(auto& t: ts) {
        t.join();
    }

    params.elapsed = timer.elapsed_reset();

    if (params.print_time) {
        for(auto& s: stats) { info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots); }
        info("n_threads = {}, chunk = {}, elapsed = {} sec", n_threads, params.chunk_size, params.elapsed);
        std::cerr << "n_threads = " << n_threads << ", elapsed = " << params.elapsed << std::endl;
    }

    // write reduced matrix back, collect V matrix, mark as reduced
    for(size_t i = 0; i < n_cols; ++i) {
        auto p = r_v_matrix[i].load(std::memory_order_relaxed);
        r_data[i] = std::move(p->r_column);
        v_data[i] = std::move(p->v_column);
    }

//    IC("After R writen back: ", d_data);

    is_reduced = true;
}



template<class Int>
template<class Real, class L>
Diagrams<Real> VRUDecomposition<Int>::diagram(const Filtration<Int, Real, L>& fil, bool include_inf_points) const
{
    if (not is_reduced)
        throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

    Diagrams<Real> result(fil.max_dim());

    std::unordered_set<Int> rows_with_lowest_one;

    if (include_inf_points)
        for(size_t i = 0; i < r_data.size(); ++i)
            if (!is_zero(&r_data[i]))
                rows_with_lowest_one.insert(low(&r_data[i]));

    for(size_t col_idx = 0; col_idx < r_data.size(); ++col_idx) {
        auto col = &r_data[col_idx];

        auto simplex_idx = fil.index_in_filtration(col_idx, dualize());

        if (is_zero(col)) {
            if (not include_inf_points or rows_with_lowest_one.count(col_idx) != 0)
                // we don't want infinite points or col_idx is a negative simplex
                continue;

            // point at infinity
            dim_type dim = fil.dim_by_sorted_id(simplex_idx);
            Real birth = fil.value_by_sorted_id(simplex_idx);
            Real death = fil.infinity();

            result.add_point(dim, birth, death);
        } else {
            // finite point
            Int birth_idx = fil.index_in_filtration(low(col), dualize()), death_idx = simplex_idx;
            dim_type dim = fil.dim_by_sorted_id(birth_idx);
            Real birth = fil.value_by_sorted_id(birth_idx), death = fil.value_by_sorted_id(death_idx);

            if (birth != death)
                result.add_point(dim, birth, death);
        }
    }
    return result;
}

template<class Int>
template<class Real, class L>
Diagrams<size_t> VRUDecomposition<Int>::index_diagram(const Filtration<Int, Real, L>& fil, bool include_inf_points, bool include_zero_persistence_points) const
{
    if (not is_reduced)
        throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

    Diagrams<size_t> result(fil.max_dim());

    std::unordered_set<size_t> rows_with_lowest_one;

    constexpr size_t plus_inf = std::numeric_limits<size_t>::max();

    if (include_inf_points)
        for(size_t i = 0; i < r_data.size(); ++i)
            if (!is_zero(&r_data[i]))
                rows_with_lowest_one.insert(low(&r_data[i]));

    for(size_t col_idx = 0; col_idx < r_data.size(); ++col_idx) {
        auto col = &r_data[col_idx];

        if (is_zero(col)) {
            if (!include_inf_points or rows_with_lowest_one.count(col_idx) != 0)
                continue;

            dim_type dim = fil.dim_by_sorted_id(col_idx);

            result.add_point(dim, col_idx, plus_inf);
        } else {
            // finite point
//            fil.
            size_t birth_idx = static_cast<size_t>(low(col));
            size_t death_idx = col_idx;

            dim_type dim = fil.dim_by_sorted_id(birth_idx);

            if (include_zero_persistence_points or fil.value_by_sorted_id(birth_idx) != fil.value_by_sorted_id(death_idx))
                result.add_point(dim, birth_idx, death_idx);
        }
    }

    return result;
}



template<typename Int>
std::ostream& operator<<(std::ostream& out, const VRUDecomposition<Int>& m)
{
     out << "Matrix D[\n";
    for(size_t col_idx = 0; col_idx < m.r_data.size(); ++col_idx) {
        out << "Column " << col_idx << ": ";
        for(const auto& x: m.d_data[col_idx])
            out << x << " ";
        out << "\n";
    }
    out << "]\n";

    out << "Matrix R[\n";
    for(size_t col_idx = 0; col_idx < m.r_data.size(); ++col_idx) {
        out << "Column " << col_idx << ": ";
        for(const auto& x: m.r_data[col_idx])
            out << x << " ";
        out << "\n";
    }
    out << "]\n";

    out << "Matrix V[\n";
    for(size_t col_idx = 0; col_idx < m.v_data.size(); ++col_idx) {
        out << "Column " << col_idx << ": ";
        for(const auto& x: m.v_data[col_idx])
            out << x << " ";
        out << "\n";
    }
    out << "]\n";

    out << "Matrix U[\n";
    for(size_t row_idx = 0; row_idx < m.u_data_t.size(); ++row_idx) {
        out << "Row " << row_idx << ": ";
        for(const auto& x: m.u_data_t[row_idx])
            out << x << " ";
        out << "\n";
    }
    out << "]\n";

    return out;
}
}
