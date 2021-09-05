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
#include <unordered_set>
#include <cstdlib>
#include <stdexcept>

#include "common_defs.h"
#include "diagram.h"
#include "mem_reclamation.h"

namespace oineus {

template<typename Int_, typename Real_, typename L_>
class Filtration;

// contains indices of non-zero entries
template<typename Int>
using SparseColumn = std::vector<Int>;

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
};

template<typename IdxType>
void add_rv_column(const RVColumns<IdxType>* col_a, const RVColumns<IdxType>* col_b, RVColumns<IdxType>* sum)
{
    add_column(col_a->r_column, col_b->r_column, sum->r_column);
    add_column(col_a->v_column, col_b->v_column, sum->v_column);
}

// TODO: clean up declaration - move to matrix?
template<class RVMatrices, class AtomicIdxVector, class Int, class MemoryReclaimC>
void parallel_reduction(RVMatrices& rv, AtomicIdxVector& pivots, std::atomic<Int>& next_free_chunk,
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
                } while(pivot_idx >= 0 && pivot_r_v_column->low() != current_low);

                if (pivot_idx == -1) {
                    if (pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                        break;
                    }
                } else if (pivot_idx < current_column_idx) {
                    // pivot to the left: kill lowest one in current column
                    add_rv_column(current_r_v_column, pivot_r_v_column, reduced_r_v_column.get());

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
struct SparseMatrix {
    // types
    using Int = Int_;
    using IntSparseColumn = SparseColumn<Int>;
    using MatrixData = std::vector<IntSparseColumn>;
    using AtomicIdxVector = std::vector<std::atomic<Int>>;

    // data
    MatrixData data;
    MatrixData v_data;
    bool is_reduced {false};

    // methods

    [[nodiscard]] size_t size() const { return data.size(); }

    void append(SparseMatrix&& other);

    void reduce_parallel(Params& params);

    template<typename Real, typename L>
    Diagrams<Real> diagram(const Filtration<Int, Real, L>& fil) const;

    template<typename Real, typename L>
    Diagrams<size_t> index_diagram_finite(const Filtration<Int, Real, L>& fil) const;


    template<typename Int>
    friend std::ostream& operator<<(std::ostream& out, const SparseMatrix<Int>& m);
};

template<class Int>
void SparseMatrix<Int>::append(SparseMatrix&& other)
{
    data.insert(data.end(),
            std::make_move_iterator(other.data.begin()),
            std::make_move_iterator(other.data.end()));
}

template<class Int>
void SparseMatrix<Int>::reduce_parallel(Params& params)
{
    using namespace std::placeholders;

    size_t n_cols = size();

    v_data = std::vector<IntSparseColumn>(n_cols);

    using RVColumnC = RVColumns<Int>;
    using APRVColumn = std::atomic<RVColumnC*>;
    using RVMatrix = std::vector<APRVColumn>;
    using MemoryReclaimC = MemoryReclaim<RVColumnC>;

    RVMatrix r_v_matrix(n_cols);

    // move data to r_v_matrix
    for(size_t i = 0; i < n_cols; ++i) {
        IntSparseColumn v_column = {static_cast<Int>(i)};
        r_v_matrix[i] = new RVColumnC(std::move(data[i]), std::move(v_column));
    }
    debug("Matrix moved");

    std::atomic<Int> counter;
    counter = 0;

    std::atomic<Int> next_free_chunk;

    AtomicIdxVector pivots(n_cols);
    for(auto& p : pivots) {
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
                std::ref(r_v_matrix), std::ref(pivots), std::ref(next_free_chunk),
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

    for(auto& t : ts) {
        t.join();
    }

    params.elapsed = timer.elapsed_reset();

    if (params.print_time) {
        for(auto& s : stats) { info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots); }
        info("n_threads = {}, chunk = {}, elapsed = {} sec", n_threads, params.chunk_size, params.elapsed);
    }

    // write reduced matrix back, collect V matrix, mark as reduced
    for(size_t i = 0; i < n_cols; ++i) {
        auto p = r_v_matrix[i].load(std::memory_order_relaxed);
        data[i] = std::move(p->r_column);
        v_data[i] = std::move(p->v_column);
    }

    is_reduced = true;
}

template<class Int>
template<class Real, class L>
Diagrams<Real> SparseMatrix<Int>::diagram(const Filtration<Int, Real, L>& fil) const
{
    if (not is_reduced)
        throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

    Diagrams<Real> result;

    std::unordered_set<Int> rows_with_lowest_one;

    for(size_t i = 0; i < data.size(); ++i) {
        if (!is_zero(&data[i]))
            rows_with_lowest_one.insert(low(&data[i]));
    }

    for(size_t col_idx = 0; col_idx < data.size(); ++col_idx) {
        auto col = &data[col_idx];

        if (is_zero(col)) {
            if (rows_with_lowest_one.count(col_idx) == 0) {
                // point at infinity

                dim_type dim = fil.dim_by_sorted_id(col_idx);
                Real birth = fil.value_by_sorted_id(col_idx);
                Real death = std::numeric_limits<Real>::infinity();

                result.add_point(dim, birth, death);
            }
        } else {
            // finite point
            Int birth_idx = low(col);
            Int death_idx = col_idx;

            dim_type dim = fil.dim_by_sorted_id(birth_idx);

            Real birth = fil.value_by_sorted_id(birth_idx);
            Real death = fil.value_by_sorted_id(death_idx);

            if (birth != death)
                result.add_point(dim, birth, death);
        }
    }

    return result;
}

template<class Int>
template<class Real, class L>
Diagrams<size_t> SparseMatrix<Int>::index_diagram_finite(const Filtration<Int, Real, L>& fil) const
{
    if (not is_reduced)
        throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

    Diagrams<size_t> result;

    for(size_t death_idx = 0; death_idx < data.size(); ++death_idx)
        if (not is_zero(data[death_idx])) {
            size_t birth_idx = static_cast<size_t>(low(data[death_idx]));
            dim_type dim = fil.dim_by_sorted_id(birth_idx);
            result.add_point(dim, birth_idx, death_idx);
        }

    return result;
}

template<typename Int>
std::ostream& operator<<(std::ostream& out, const SparseMatrix<Int>& m)
{
    out << "Matrix[\n";
    for(size_t col_idx = 0; col_idx < m.data.size(); ++col_idx) {
        out << "Column " << col_idx << ": ";
        for(const auto& x : m.data[col_idx])
            out << x << " ";
        out << "\n";
    }
    out << "]\n";
    return out;
}
}
