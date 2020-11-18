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
#include <chrono>
#include <stdlib.h>

#include "common_defs.h"
#include "mem_reclamation.h"


namespace oineus {


    template<typename Int_, typename Real_>
    class Filtration;

    template<typename Real_>
    class Diagram;


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
    bool is_zero(const SparseColumn<IdxType>* c)
    {
        return c->empty();
    }

    template<typename IdxType>
    void add_column(const SparseColumn<IdxType>* col_a, const SparseColumn<IdxType>* col_b, SparseColumn<IdxType>* sum)
    {
        auto a_iter = col_a->cbegin();
        auto b_iter = col_b->cbegin();

        sum->clear();

        while(true) {
            if (a_iter == col_a->cend() && b_iter == col_b->cend()) {
                break;
            } else if (a_iter == col_a->cend() && b_iter != col_b->cend()) {
                sum->push_back(*b_iter++);
            } else if (a_iter != col_a->cend() && b_iter == col_b->cend()) {
                sum->push_back(*a_iter++);
            } else if (*a_iter < *b_iter) {
                sum->push_back(*a_iter++);
            } else if (*b_iter < *a_iter) {
                sum->push_back(*b_iter++);
            } else {
                assert(*a_iter == *b_iter);
                ++a_iter;
                ++b_iter;
            }
        }
    }


    // TODO: clean up declaration - move to matrix?
    template<class APSparseMatrix, class AtomicIdxVector, class Int, class MemoryReclaimC>
    void parallel_reduction(APSparseMatrix& r, AtomicIdxVector& pivots, std::atomic<Int>& next_free_chunk,
            const Params params, int thread_idx, MemoryReclaimC* mm, ThreadStats& stats, bool go_down)
    {
        using IntSparseColumn = SparseColumn<Int>;
        using PSparseColumn = IntSparseColumn*;

        std::memory_order acq = params.acq_rel ? std::memory_order_acquire : std::memory_order_seq_cst;
        std::memory_order rel = params.acq_rel ? std::memory_order_release : std::memory_order_seq_cst;
        std::memory_order relax = params.acq_rel ? std::memory_order_relaxed : std::memory_order_seq_cst;

        debug("thread {} started, mm = {}", thread_idx, (void*) (mm));
        std::unique_ptr<IntSparseColumn> reduced_column(new IntSparseColumn);
        std::unique_ptr<IntSparseColumn> reduced_column_final(new IntSparseColumn);

        const int n_cols = r.size();

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

                PSparseColumn current_column = r[current_column_idx].load(acq);
                PSparseColumn current_column_orig = current_column;

                bool update_column = false;

                int pivot_idx;

                if (params.clearing_opt) {
                    if (!is_zero(current_column)) {
                        int c_pivot_idx = pivots[current_column_idx].load(acq);
                        if (c_pivot_idx >= 0) {
                            // unset pivot from current_column_idx, if necessary
                            int c_current_low = low(current_column);
                            int c_current_column_idx = current_column_idx;
                            pivots[c_current_low].compare_exchange_weak(c_current_column_idx, -1, rel, relax);

                            // zero current column
                            current_column = new IntSparseColumn();
                            r[current_column_idx].store(current_column, rel);
                            mm->retire(current_column_orig);
                            current_column_orig = current_column;

                            stats.n_cleared++;
                        }
                    }
                }

                while(!is_zero(current_column)) {

                    int current_low = low(current_column);
                    PSparseColumn pivot_column = nullptr;

                    debug("thread {}, column = {}, low = {}", thread_idx, current_column_idx, current_low);

                    do {
                        pivot_idx = pivots[current_low].load(acq);
                        if (pivot_idx >= 0) {
                            pivot_column = r[pivot_idx].load(acq);
                        }
                    }
                    while(pivot_idx >= 0 && low(pivot_column) != current_low);

                    if (pivot_idx == -1) {
                        if (pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                            break;
                        }
                    } else if (pivot_idx < current_column_idx) {
                        // pivot to the left: kill lowest one in current column
                        add_column(current_column, pivot_column, reduced_column.get());
                        update_column = true;
                        reduced_column_final->swap(*reduced_column);
                        current_column = reduced_column_final.get();
                    } else if (pivot_idx > current_column_idx) {

                        stats.n_right_pivots++;

                        // pivot to the right: switch to reducing r[pivot_idx]
                        if (update_column) {
                            // write copy of reduced column into matrix
                            current_column = new IntSparseColumn(reduced_column_final->begin(),
                                    reduced_column_final->end());
                            r[current_column_idx].store(current_column, rel);
                            mm->retire(current_column_orig);
                            current_column_orig = current_column;
                            update_column = false;
                        }

                        // set current column as new pivot, start reducing column r[pivot_idx]
                        if (pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                            current_column_idx = pivot_idx;
                            current_column = r[current_column_idx].load(acq);
                            current_column_orig = current_column;

                        }
                    }
                } // reduction loop

                if (update_column) {
                    // write copy of reduced column to matrix
                    // TODO: why not use reduced_column_final directly?
                    current_column = new IntSparseColumn(reduced_column_final->begin(), reduced_column_final->end());
                    r[current_column_idx].store(current_column, rel);
                    mm->retire(current_column_orig);
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

        using Int = Int_;
        using IntSparseColumn = SparseColumn<Int>;

        std::vector<IntSparseColumn> data;
        std::vector<Int> column_dimensions_;

        size_t size() const { return data.size(); }

        // contains indices of non-zero entries
        using PSparseColumn = IntSparseColumn*;
        using APSparseColumn = std::atomic<IntSparseColumn*>;
        using APSparseMatrix = std::vector<APSparseColumn>;
        using MemoryReclaimC = MemoryReclaim<IntSparseColumn>;
        using AtomicIdxVector = std::vector<std::atomic<Int>>;

        void append(SparseMatrix&& other)
        {
            data.insert(data.end(),
                    std::make_move_iterator(other.data.begin()),
                    std::make_move_iterator(other.data.end()));
        }

        void reduce_parallel(Params& params)
        {

            using namespace std::placeholders;

            // copy r
            APSparseMatrix r_matrix(data.size());
            for(size_t i = 0; i < data.size(); ++i) {
                r_matrix[i] = new IntSparseColumn(data[i].cbegin(), data[i].cend());
            }

            debug("Matrix copied");

            std::atomic<Int> counter;

            counter = 0;

            std::atomic<Int> next_free_chunk;
            Int n_cols = size();

            AtomicIdxVector pivots(n_cols);
            for(auto& p : pivots) {
                p = -1;
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

            std::chrono::high_resolution_clock timer;
            auto start_time = timer.now();

            for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

                mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
                stats.emplace_back(thread_idx);

                ts.emplace_back(parallel_reduction<APSparseMatrix, AtomicIdxVector, Int, MemoryReclaimC>,
                        std::ref(r_matrix), std::ref(pivots), std::ref(next_free_chunk),
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

            std::chrono::duration<double> elapsed = timer.now() - start_time;

            params.elapsed = elapsed.count();

            if (params.print_time) {
                for(auto& s : stats) {
                    info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots);
                }

                info("n_threads = {}, chunk = {}, elapsed = {} sec", n_threads, params.chunk_size, elapsed.count());
            }

            // write reduced matrix back to r
            for(size_t i = 0; i < data.size(); ++i) {
                data[i] = *r_matrix[i];
            }
        }

        template<typename Real>
        Diagram<Real> diagram(const Filtration<Int, Real>& fil) const
        {
            Diagram<Real> result;

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

        template<typename Int>
        friend std::ostream& operator<<(std::ostream& out, const SparseMatrix<Int>& m);
    };

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
