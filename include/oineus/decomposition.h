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

#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

#include "common_defs.h"
#include "timer.h"
#include "diagram.h"
#include "mem_reclamation.h"
#include "sparse_matrix.h"

namespace oineus {

    template<typename Cell, typename Real>
    class Filtration;

    using Idx = int;
    using AtomicIdxVector = std::vector<std::atomic<Idx>>;

    template<class MatrixTraits, class MemoryReclaimC>
    void update_column(bool& needs_update,
            MemoryReclaimC* mm,
            typename MatrixTraits::CachedColumn& cached_reduced_col,
            typename MatrixTraits::PColumn orig_col,
            typename MatrixTraits::AMatrix& rv,
            Idx current_column_idx)
    {
        if (needs_update) {
            // orig_col and new_col can be nullptr for zero columns, other functions can handle that
            typename MatrixTraits::PColumn new_col = MatrixTraits::load_from_cache(cached_reduced_col);

            assert(MatrixTraits::check_col_duplicates(new_col).empty());

            rv[current_column_idx].store(new_col, std::memory_order_seq_cst);

            mm->retire(orig_col);
        }
        needs_update = false;
    }

    template<class Int, class PLogger>
    void get_next_chunk(const bool clearing, std::atomic<int>& next_free_chunk, const int chunk_size, const int n_cols,
                        int&my_chunk, int& chunk_begin, int& chunk_end, bool& done, PLogger& logger,
                        // these are only used if clearing is true
                        std::vector<std::atomic<int>>& next_free_chunks, const std::vector<Int>& dim_first, const std::vector<Int>& dim_last, Int& current_dim)
    {
        static_assert(std::is_signed_v<Int>, "Int must be signed: current_dim can go below 0 to signal done");
        done = false;
        if (clearing) {
            while(true) {
                if (current_dim < 0) {
                    done = true;
                    return;
                }
                my_chunk = next_free_chunks[current_dim].fetch_add(1, std::memory_order_seq_cst);
                chunk_begin = dim_first.at(current_dim) + my_chunk * chunk_size;
                if (chunk_begin > dim_last.at(current_dim)) {
                    current_dim--;
                    continue;
                }
                chunk_end = std::min(dim_last.at(current_dim) + 1, dim_first.at(current_dim) + (my_chunk + 1) * chunk_size);
                logger->debug("exiting get_next_chunk, clearing = {}, done = {}, my_chunk = {}, chunk_begin = {}, chunk_end = {}, current_dim ={}", clearing, done, my_chunk, chunk_begin, chunk_end, current_dim);
                break;
            }
        } else {
            my_chunk = next_free_chunk.fetch_add(1, std::memory_order_seq_cst);
            chunk_begin = my_chunk * chunk_size;
            chunk_end = std::min(n_cols, (my_chunk + 1) * chunk_size);
            done = (chunk_begin >= n_cols || chunk_end <= 0);
            logger->debug("exiting get_next_chunk, clearing = {}, done = {}, my_chunk = {}, chunk_begin = {}, chunk_end = {}", clearing, done, my_chunk, chunk_begin, chunk_end);
        }
    }

    template<class MatrixTraits, class Int, class MemoryReclaimC>
    void parallel_reduction(typename MatrixTraits::AMatrix& rv, AtomicIdxVector& pivots, std::atomic<int>& next_free_chunk,
            const Params params, int thread_idx, MemoryReclaimC* mm, ThreadStats& stats,
            std::vector<std::atomic<int>>& next_free_chunks, std::vector<Int>& dim_first, std::vector<Int>& dim_last)
    {
        // set logger
        bool log_each_thread_to_file = true;
        std::shared_ptr<spd::logger> logger;
        std::string logger_name = "oineus_log_thread_" + std::to_string(thread_idx);
        logger = spd::get(logger_name);
        if (!logger) {
            if (log_each_thread_to_file)
                logger = spd::basic_logger_mt(logger_name.c_str(), logger_name.c_str());
            else
                logger = spd::stderr_color_mt(logger_name.c_str());
        }
        logger->set_level(params.spdlog_level);
        logger->flush_on(params.spdlog_level);

        using PColumn = typename MatrixTraits::PColumn;

        std::memory_order acq = std::memory_order_seq_cst;
        std::memory_order rel =  std::memory_order_seq_cst;
        std::memory_order relax = std::memory_order_seq_cst;

        logger->debug("thread {} started, mm = {}, next_free_chunk = {}", thread_idx, (void*) (mm), (void*)(&next_free_chunk));

        const int n_cols = rv.size();

        int my_chunk, chunk_begin, chunk_end;
        Int current_dim = params.clearing_opt ? dim_first.size() - 1 : 0;
        bool done;

        do {
            get_next_chunk(params.clearing_opt, next_free_chunk, params.chunk_size, n_cols,
                        my_chunk, chunk_begin, chunk_end, done, logger,
                        // the following are only used if clearing is true
                        next_free_chunks, dim_first, dim_last, current_dim);

            if (done) {
                logger->debug("Thread {} finished", thread_idx);
                break;
            }

            logger->debug("thread {}, processing chunk {}, from {} to {}, n_cols = {}, current_dim = {}", thread_idx, my_chunk, chunk_begin, chunk_end, n_cols, current_dim);

            Idx current_column_idx = chunk_begin;
            int next_column = current_column_idx + 1;

#ifndef NDEBUG
            std::set<Idx> unprocessed_cols;
            for(Idx ii = chunk_begin; ii < chunk_end; ++ii)
                unprocessed_cols.insert(ii);
#endif
            while(true) {
                logger->debug("thread {}, started reducing column = {}", thread_idx, current_column_idx);

                PColumn orig_col = rv[current_column_idx].load(acq);

                if (orig_col == nullptr) {
                    // this column has already been zeroed by someone else, continue to next one
                    current_column_idx = next_column;
                    next_column = current_column_idx + 1;

                    logger->debug("current column is already 0, advancing to next in chunk, current_column_idx = {}, next_column = {}", current_column_idx, next_column);

                    if (current_column_idx < chunk_end)
                        continue;
                    else
                        break;
                }

                assert(MatrixTraits::check_col_duplicates(orig_col).empty());

                auto cached_reduced_col = MatrixTraits::load_to_cache(orig_col);

#ifndef NDEBUG
                unprocessed_cols.erase(current_column_idx);
#endif

                if (params.clearing_opt) {
                    if (!MatrixTraits::is_zero(cached_reduced_col)) {
                        int c_pivot_idx = pivots[current_column_idx].load(acq);
                        if (c_pivot_idx >= 0) {
                            // unset pivot from current_column_idx, if necessary
                            int c_current_low = MatrixTraits::low(cached_reduced_col);
                            Idx c_current_column_idx = current_column_idx;

                            pivots[c_current_low].compare_exchange_strong(c_current_column_idx, -1, rel, relax);
                            // if CAS fails here, it's totally fine, just means
                            // that this column is not set as pivot

                            rv[current_column_idx].store(nullptr, rel);
                            mm->retire(orig_col);

                            stats.n_cleared++;

                            current_column_idx = next_column;
                            next_column = current_column_idx + 1;

                            logger->debug("Cleared column, advanced to next in chunk, current_column_idx = {}, next_column = {}", current_column_idx, next_column);

                            if (current_column_idx < chunk_end)
                                continue;
                            else
                                break;
                        }
                    }
                }

                bool needs_update = false;
                bool start_over = false;

                Idx pivot_idx;

                while(!MatrixTraits::is_zero(cached_reduced_col)) {

                    auto current_low = MatrixTraits::low(cached_reduced_col);
                    PColumn pivot_col = nullptr;

                    logger->debug("thread {}, column = {}, low = {}", thread_idx, current_column_idx, current_low);

                    do {
                        pivot_idx = pivots[current_low].load(acq);
                        if (pivot_idx >= 0) {
                            pivot_col = rv[pivot_idx].load(acq);
                        }
//                      logger->trace("thread {}, column = {}, low = {}, pivot_idx = {}, pivolt_col = {}", thread_idx, current_column_idx, current_low, pivot_idx, (void*) pivot_col);
                    } while(pivot_idx >= 0 && pivot_col != nullptr && MatrixTraits::low(pivot_col) != current_low);

                    logger->debug("thread {}, column = {}, loaded pivot column, pivot_idx = {}", thread_idx, current_column_idx, pivot_idx);

                    if (pivot_idx == -1) {

                        update_column<MatrixTraits, MemoryReclaimC>(needs_update, mm, cached_reduced_col, orig_col, rv, current_column_idx);

                        if (!pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                            start_over = true;
                        }
                        logger->debug("thread {}, column = {}, after CAS to set myself as pivot start_over = {}", thread_idx, current_column_idx, start_over);
                        break;
                    } else if (pivot_idx < current_column_idx) {
                        // for now, record statistics for r matrix only
#ifdef OINEUS_GATHER_ADD_STATS
                        stats.r_column_summand_sizes[{MatrixTraits::r_column_size(*pivot_col), MatrixTraits::r_column_size(cached_reduced_col)}]++;
                        stats.v_column_summand_sizes[{MatrixTraits::v_column_size(*pivot_col), MatrixTraits::v_column_size(cached_reduced_col)}]++;
#endif
                        // pivot to the left: kill lowest one in current column
                        MatrixTraits::add_to_cached(pivot_col, cached_reduced_col);
                        logger->debug("thread {}, column = {}, added pivot to the left OK", thread_idx, current_column_idx);
                        needs_update = true;
                    } else if (pivot_idx > current_column_idx) {
                        // pivot to the right: switch to reducing r[pivot_idx]

                        logger->debug("Pivot to the right, current_column_idx = {}, next_column = {}, pivot_idx = {}", current_column_idx, next_column, pivot_idx);
                        stats.n_right_pivots++;

                        // store current column, if needed
                        update_column<MatrixTraits, MemoryReclaimC>(needs_update, mm, cached_reduced_col, orig_col, rv, current_column_idx);

                        // set current column as new pivot, start reducing column r[pivot_idx]
                        if (pivots[current_low].compare_exchange_weak(pivot_idx, current_column_idx, rel, relax)) {
                            current_column_idx = pivot_idx;
                            orig_col = rv[current_column_idx].load(acq);
                            cached_reduced_col = MatrixTraits::load_to_cache(orig_col);
                            logger->debug("Pivot to the right, CAS okay, set current_column_idx = {}, next_column = {}", current_column_idx, next_column);
                        } else {
                            logger->debug("Pivot to the right, CAS failed, set start_over = TRUE");
                            start_over = true;
                            break;
                        }
                    }
                } // reduction loop

                logger->debug("Exited reduction loop, needs_update = {}, start_over = {}", needs_update, start_over);

                if (not start_over) {
                    update_column<MatrixTraits, MemoryReclaimC>(needs_update, mm, cached_reduced_col, orig_col, rv, current_column_idx);
                    current_column_idx = next_column;
                    next_column = current_column_idx + 1;
                    logger->debug("not starting over, advanced to next column, current_column_idx = {}, next_column = {}, chunk_end = {}", current_column_idx, next_column, chunk_end);
                    if (current_column_idx >= chunk_end) {
                        logger->debug("exiting loop over chunk columns");
                        break;
                    }
                } // else we re-start with the same current_column_idx re-reading
                // the column, because one of CAS operations failed

            } //loop over columns

#ifndef NDEBUG
            if (!unprocessed_cols.empty()) {
                logger->critical("Error: chunk_begin = {}, {} unprocessed_cols remaining, first: {}, stats = {}", chunk_begin, unprocessed_cols.size(), *unprocessed_cols.begin(), stats);
                logger->flush();
                throw std::runtime_error("some columns in chunk not processed");
            }
#endif
#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT_WITH_GIL
            if (my_chunk % 100 == 0) {
                logger->debug("checking for Ctrl-C in Python...");
                OINEUS_CHECK_FOR_PYTHON_INTERRUPT_WITH_GIL;
                logger->debug("checking for Ctrl-C in Python... done");
            }
#endif
            mm->quiescent();
        } while(true); // loop over chunks
        logger->debug("thread {}, EXIT reduction, stats = {}", thread_idx, stats);
    }

    template<typename Int_>
    struct VRUDecomposition {
        // types
        using Int = Int_;
        using IntSparseColumn = std::vector<Int>;
        using MatrixData = std::vector<IntSparseColumn>;

        // data
        MatrixData d_data;
        MatrixData r_data;
        MatrixData v_data;
        MatrixData u_data_t;
        bool is_reduced {false};
        bool dualize_ {false};

        std::vector<Int> dim_first;
        std::vector<Int> dim_last;

        size_t n_rows {0};

        // methods
        VRUDecomposition() = default;
        VRUDecomposition(const VRUDecomposition&) = default;
        VRUDecomposition(VRUDecomposition&&) noexcept = default;
        VRUDecomposition& operator=(VRUDecomposition&&) noexcept = default;
        VRUDecomposition& operator=(const VRUDecomposition&) = default;

        template<class C, class R>
        VRUDecomposition(const Filtration<C, R>& fil, bool _dualize, int n_threads=8)
                :
                d_data(_dualize ? fil.coboundary_matrix(n_threads) : fil.boundary_matrix(n_threads)),
                r_data(d_data),
                dualize_(_dualize),
                dim_first(fil.dim_first()),
                dim_last(fil.dim_last()),
                n_rows(d_data.size())
        {
            if (dualize_) {
                std::reverse(dim_first.begin(), dim_first.end());
                std::reverse(dim_last.begin(), dim_last.end());
                std::vector<Int> new_dim_first, new_dim_last;
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

        VRUDecomposition(const MatrixData& d, size_t n_rows = std::numeric_limits<decltype(n_rows)>::max(), bool dualize=false)
                :
                d_data(d),
                r_data(d),
                dualize_(dualize),
                // TODO: think about dimensions here
                dim_first(std::vector<Int>({0})),
                dim_last(std::vector<Int>({static_cast<Int>(d.size() - 1)})),
                n_rows(n_rows == std::numeric_limits<decltype(n_rows)>::max() ? d_data.size() : n_rows)
        {
        }

        [[nodiscard]] size_t size() const { return r_data.size(); }

        bool dualize() const { return dualize_; }

        void reduce(Params& params);

        void reduce_serial(Params& params);

        void reduce_parallel_r_only(Params& params);
        void reduce_parallel_rv(Params& params);

        bool is_negative(size_t simplex) const
        {
            return not is_positive(simplex);
        }

        bool is_positive(size_t simplex) const
        {
            assert(is_reduced);
            return is_zero(r_data[simplex]);
        }

        bool has_matrix_u() const { return u_data_t.size() > 0; }
        bool has_matrix_v() const { return v_data.size() > 0; }

        template<typename Cell, typename Real>
        Diagrams<Real> diagram_general(const Filtration<Cell, Real>& fil, bool include_all, bool include_inf_points, bool only_zero_persistence) const;

        template<typename Cell, typename Real>
        Diagrams<Real> diagram_general(const Filtration<Cell, Real>& fil,
                const typename Cell::UidSet& relative,
                bool include_all, bool include_inf_points, bool only_zero_persistence) const;

        template<typename Cell, typename Real>
        Diagrams<Real> diagram(const Filtration<Cell, Real>& fil, bool include_inf_points) const;

        template<typename Cell, typename Real>
        Diagrams<Real> diagram(const Filtration<Cell, Real>& fil, const typename Cell::UidSet& relative, bool include_inf_points) const;

        template<typename Cell, typename Real>
        Diagrams<Real> zero_persistence_diagram(const Filtration<Cell, Real>& fil) const;

        template<typename Int>
        friend std::ostream& operator<<(std::ostream& out, const VRUDecomposition<Int>& m);

        bool sanity_check();

        const MatrixData& get_D() const
        {
            return d_data;
        }

        const MatrixData& get_V() const
        {
            return v_data;
        }

        const MatrixData& get_R() const
        {
            return r_data;
        }

        size_t filtration_index(size_t matrix_idx) const
        {
            return dualize() ? size() - matrix_idx - 1 : matrix_idx;
        }

        bool is_R_column_zero(size_t col_idx) const { return r_data[col_idx].empty(); }
        bool is_V_column_zero(size_t col_idx) const { return v_data[col_idx].empty(); }
    };

    template<class Int>
    bool are_matrix_columns_sorted(const std::vector<std::vector<Int>>& matrix_cols)
    {
        for(auto&& col: matrix_cols) {
            if (not std::is_sorted(col.begin(), col.end())) {
                for(auto e: col)
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

        for(auto&& col: matrix_cols) {
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
            for(auto row_idx: matrix_cols[col_idx])
                e_cols.emplace(static_cast<size_t>(row_idx), col_idx);

        for(size_t row_idx = 0; row_idx < matrix_rows.size(); ++row_idx)
            for(auto col_idx: matrix_rows[row_idx])
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
        if (has_matrix_u() and not are_matrix_columns_sorted(u_data_t)) {
            std::cerr << "sanity_check: U not sorted!" << std::endl;
            return false;
        }
        if (verbose) std::cerr << "U sorted" << std::endl;

        size_t n_simplices = r_data.size();

        // all matrices are square
        SparseMatrix<Int> dd(d_data, n_simplices, true);
        SparseMatrix<Int> rr(r_data, n_simplices, true);
        SparseMatrix<Int> uu(u_data_t, n_simplices, false);
        SparseMatrix<Int> vv(v_data, n_simplices, true);

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

        if (has_matrix_u() and not uu.is_upper_triangular()) {
            std::cerr << "U not upper-triangular" << std::endl;
            return false;
        }
        if (verbose) std::cerr << "U upper-triangular" << std::endl;

        dv.compute_cols();

        for(size_t i = 0; i < rr.n_cols(); ++ i) {
            if (not rr.is_col_zero(i) and rr.col(i) != dv.col(i)) {
                std::cerr << "R = " << rr << std::endl;
                std::cerr << "D = " << dd << std::endl;
                std::cerr << "V = " << vv << std::endl;
                std::cerr << "U = " << uu << std::endl;
                std::cerr << "R != DV" << std::endl;
                return false;
            }
        }

        if (verbose) std::cerr << "R = DV" << std::endl;


        //if (uv != ii) {
        //    std::cerr << "uv: " << uv.sanity_check() << std::endl;
        //    std::cerr << "ii: " << ii.sanity_check() << std::endl;
        //    std::cerr << "UV != I" << std::endl;
        //    return false;
        //}
        //if (verbose) std::cerr << "UV = I" << std::endl;
        return true;
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;
        if (d_data.empty()) {
            is_reduced = true;
            return;
        }

        if (params.n_threads > 1 and params.compute_u)
            throw std::runtime_error("Cannot compute U matrix in parallel");

        if (params.n_threads == 1)
            reduce_serial(params);
        else if (params.compute_v)
            reduce_parallel_rv(params);
        else
            reduce_parallel_r_only(params);
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce_serial(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;

        Timer timer_total;

        using MatrixTraits = SimpleSparseMatrixTraits<Int, 2>;

        ThreadStats stats {0};
        int n_cleared = 0;

        std::unordered_set<Int> cleared_cols;

        if (params.compute_v)
            v_data = MatrixTraits::eye(d_data.size());

        if (params.compute_u)
            u_data_t = MatrixTraits::eye(d_data.size());

        // D matrix empty, nothing to do
        if (d_data.empty())
            return;

        std::vector<Int> pivots(n_rows, -1);

        // homology: go from top dimension to 0, to make clearing possible
        // cohomology:
        IntSparseColumn new_col;
        for(int dim = dim_first.size() - 1; dim >= 0; --dim) {
            for(Int i = dim_first[dim]; i <= dim_last[dim]; ++i) {
                if (params.clearing_opt and not is_zero(r_data[i])) {
                    // simplex i is pivot -> i is positive -> its column is 0
                    if (pivots[i] >= 0) {
                        r_data[i].clear();
                        n_cleared++;
                        // U. Bauer's trick to get a valid V column for cleared columns
                        if (params.compute_v) {
                            v_data[i] = r_data[pivots[i]];
                        }
                        continue;
                    }
                }

                typename MatrixTraits::CachedColumn cached_r_col = MatrixTraits::load_to_cache(r_data[i]);
                typename MatrixTraits::CachedColumn cached_v_col;

                if (params.compute_v) {
                    cached_v_col = MatrixTraits::load_to_cache(v_data[i]);
                }

                while(not MatrixTraits::is_zero(cached_r_col)) {

                    Int& pivot = pivots[MatrixTraits::low(cached_r_col)];

                    if (pivot == -1) {
                        pivot = i;
                        break;
                    } else {

#ifdef OINEUS_GATHER_ADD_STATS
                        stats.r_column_summand_sizes[{MatrixTraits::r_column_size(r_data[pivot]), MatrixTraits::r_column_size(cached_r_col)}]++;
#endif
                        MatrixTraits::add_to_cached(r_data[pivot], cached_r_col);

                        if (params.compute_v) {
#ifdef OINEUS_GATHER_ADD_STATS
                            stats.v_column_summand_sizes[{MatrixTraits::v_column_size(v_data[pivot]), MatrixTraits::v_column_size(cached_v_col)}]++;
#endif
                            MatrixTraits::add_to_cached(v_data[pivot], cached_v_col);
                        }

                        if (params.compute_u)
                            u_data_t[pivot].push_back(i);
                    }

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                    if (i % 1000 == 0) {
                        OINEUS_CHECK_FOR_PYTHON_INTERRUPT_WITH_GIL;
                    }
#endif
                } // reduction loop
                MatrixTraits::load_from_cache(cached_r_col, r_data[i]);
                if (params.compute_v)
                    MatrixTraits::load_from_cache(cached_v_col, v_data[i]);
            } // loop over columns in fixed dimension
        } // loop over dimensions

        params.elapsed = timer_total.elapsed_reset();

        if (params.print_time or params.verbose)
            std::cerr << "reduce_serial, matrix_size = " << r_data.size() << ", clearing_opt = " << params.clearing_opt << ", n_cleared = " << n_cleared << ", total elapsed: " << params.elapsed << std::endl;

        is_reduced = true;
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce_parallel_r_only(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;
        using namespace std::placeholders;

        size_t n_cols = size();

        if (n_cols == 0)
            return;

        using MatrixTraits = SimpleSparseMatrixTraits<Int, 2>;
        using Column = typename MatrixTraits::Column;
        using AMatrix = std::vector<typename MatrixTraits::APColumn>;
        using MemoryReclaimC = MemoryReclaim<Column>;

        std::atomic<typename MemoryReclaimC::EpochCounter> counter;
        counter = 0;

        std::atomic<int> next_free_chunk = 0;
        std::vector<std::atomic<int>> next_free_chunks(dim_first.size());
        for(auto& nfc : next_free_chunks) {
            nfc.store(0, std::memory_order_relaxed);
        }

        const int n_threads = std::min(params.n_threads, std::max(1, static_cast<int>(n_cols / params.chunk_size)));

        AMatrix ar_matrix(n_cols);
        AtomicIdxVector pivots(n_rows);

        tf::Executor executor(n_threads);

        // move data to ar_matrix and set pivots in parallel
        {
            tf::Taskflow taskflow_prepare;
            taskflow_prepare.for_each_index((size_t)0, n_cols, (size_t)1,
                    [this, &ar_matrix](size_t col_idx) {
                        ar_matrix[col_idx] = new Column(std::move(r_data[col_idx]));
                        assert(MatrixTraits::check_col_duplicates(ar_matrix[col_idx]).empty());
                    });
            taskflow_prepare.for_each_index((size_t)0, n_rows, (size_t)1,
                    [&pivots](size_t col_idx) {
                        pivots[col_idx].store(-1, std::memory_order_relaxed);
                    });
            executor.run(taskflow_prepare).get();
        }

        spd::debug("Pivots initialized");


        std::vector<std::thread> ts;
        std::vector<std::unique_ptr<MemoryReclaimC>> mms;
        std::vector<ThreadStats> stats;

        mms.reserve(n_threads);
        stats.reserve(n_threads);

        Timer timer;

        for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

            mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
            stats.emplace_back(thread_idx);

            ts.emplace_back(parallel_reduction<MatrixTraits, Int, MemoryReclaimC>,
                    std::ref(ar_matrix), std::ref(pivots), std::ref(next_free_chunk),
                    params, thread_idx, mms[thread_idx].get(), std::ref(stats[thread_idx]),
                    std::ref(next_free_chunks), std::ref(dim_first), std::ref(dim_last));

#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(thread_idx, &cpuset);
            int rc = pthread_setaffinity_np(ts[thread_idx].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
#endif
        }

        spd::info("{} threads created", ts.size());

        for(auto& t: ts) {
            t.join();
        }

        params.elapsed = timer.elapsed_reset();

        if (params.print_time) {
            long total_cleared = 0;
            for(const auto& s: stats) {
                total_cleared += s.n_cleared;
                spd::info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots);
            }
            spd::info("n_threads = {}, chunk = {}, total_cleared = {}, elapsed = {} sec", n_threads, params.chunk_size, total_cleared, params.elapsed);
        }

#ifdef OINEUS_GATHER_ADD_STATS
        write_add_stats_file(stats);
#endif
        {
            tf::Taskflow taskflow_finish;
            taskflow_finish.for_each_index((size_t)0, n_cols, (size_t)1,
                    [this, &ar_matrix](size_t col_idx) {
                        auto p = ar_matrix[col_idx].load(std::memory_order_relaxed);
                        if (p) {
                            r_data[col_idx] = std::move(*p);
                            delete p;
                        } else {
                            r_data[col_idx].clear();
                        }
                    });
            executor.run(taskflow_finish).get();
        }

        is_reduced = true;
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce_parallel_rv(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;
        using namespace std::placeholders;

        size_t n_cols = size();

        if (n_cols == 0)
            return;

        int n_threads = std::min(params.n_threads, std::max(1, static_cast<int>(n_cols / params.chunk_size)));

        v_data = std::vector<IntSparseColumn>(n_cols);

        using MatrixTraits = SimpleRVMatrixTraits<Int, 2>;

        using RVColumn = typename MatrixTraits::Column;
        using RVMatrix = std::vector<typename MatrixTraits::APColumn>;
        using MemoryReclaimC = MemoryReclaim<RVColumn>;

        RVMatrix r_v_matrix(n_cols);

        std::atomic<typename MemoryReclaimC::EpochCounter> counter;
        counter = 0;

        std::atomic<int> next_free_chunk = 0;
        std::vector<std::atomic<int>> next_free_chunks(dim_first.size());
        for(auto& nfc : next_free_chunks) {
            nfc.store(0, std::memory_order_seq_cst);
        }

        AtomicIdxVector pivots(n_rows);

        tf::Executor executor(n_threads);

        // move data to ar_matrix and set pivots in parallel
        {
            tf::Taskflow taskflow_prepare;

            taskflow_prepare.for_each_index((size_t)0, n_cols, (size_t)1,
                    [this, &r_v_matrix](size_t col_idx) {
                        IntSparseColumn v_column = {static_cast<Int>(col_idx)};
                        r_v_matrix[col_idx] = new RVColumn(r_data[col_idx], v_column);
                    });
            taskflow_prepare.for_each_index((size_t)0, n_rows, (size_t)1,
                    [&pivots](size_t col_idx) {
                        pivots[col_idx].store(-1, std::memory_order_relaxed);
                    });
            executor.run(taskflow_prepare).get();
        }

        std::vector<std::thread> ts;
        std::vector<std::unique_ptr<MemoryReclaimC>> mms;
        std::vector<ThreadStats> stats;

        mms.reserve(n_threads);
        stats.reserve(n_threads);

        next_free_chunk = 0;

        Timer timer;

        for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

            mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
            stats.emplace_back(thread_idx);

            ts.emplace_back(parallel_reduction<MatrixTraits, Int, MemoryReclaimC>,
                    std::ref(r_v_matrix), std::ref(pivots), std::ref(next_free_chunk),
                    params, thread_idx, mms[thread_idx].get(), std::ref(stats[thread_idx]),
                    std::ref(next_free_chunks), std::ref(dim_first), std::ref(dim_last));

#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(thread_idx, &cpuset);
            int rc = pthread_setaffinity_np(ts[thread_idx].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
#endif
        }

        spd::info("{} threads created", ts.size());

        for(auto& t: ts) {
            t.join();
        }

        params.elapsed = timer.elapsed_reset();

        if (params.print_time) {
            long total_cleared = 0;
            for(const auto& s: stats) {
                total_cleared += s.n_cleared;
                spd::info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots);
            }
            spd::info("n_threads = {}, chunk = {}, total_cleared = {}, elapsed = {} sec", n_threads, params.chunk_size, total_cleared, params.elapsed);
        }

#ifdef OINEUS_GATHER_ADD_STATS
        write_add_stats_file(stats);
#endif
        {
            tf::Taskflow taskflow_finish;
            taskflow_finish.for_each_index((size_t)0, n_cols, (size_t)1,
                    [this, &r_v_matrix](size_t col_idx) {
                        auto p = r_v_matrix[col_idx].load(std::memory_order_relaxed);
                        if (p) {
                            r_data[col_idx] = std::move(p->r_column);
                            v_data[col_idx] = std::move(p->v_column);
                            delete p;
                        } else {
                            r_data[col_idx].clear();
                            // TODO: Bauer's trick with filling V
                            v_data[col_idx].clear();
                        }
                    });
            executor.run(taskflow_finish).get();
        }

        is_reduced = true;
    }

    template<class Int>
    template<class Cell, class Real>
    Diagrams<Real> VRUDecomposition<Int>::diagram_general(const Filtration<Cell, Real>& fil, bool include_all, bool include_inf_points, bool only_zero_persistence) const
    {
        if (not is_reduced)
            throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

        Diagrams<Real> result(fil.max_dim());

        std::unordered_set<Int> rows_with_lowest_one;

        if (include_all) {
            include_inf_points = true;
            only_zero_persistence = false;
        } else if (only_zero_persistence) {
            include_inf_points = false;
        }

        if (include_inf_points)
            for(size_t i = 0; i < r_data.size(); ++i) {
                if (!is_zero(&r_data[i]))
                    rows_with_lowest_one.insert(low(&r_data[i]));

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                if (i % 100 == 0) {
                    OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
                }
#endif

            }

        for(size_t col_idx = 0; col_idx < r_data.size(); ++col_idx) {
            auto col = &r_data[col_idx];

            auto simplex_idx = fil.index_in_filtration(col_idx, dualize());
			auto simplex_idx_us = fil.get_id_by_sorted_id(simplex_idx);
            if (is_zero(col)) {
                if (not include_inf_points or rows_with_lowest_one.count(col_idx) != 0)
                    // we don't want infinite points or col_idx is a negative simplex
                    continue;

                // point at infinity
                dim_type dim = fil.dim_by_sorted_id(simplex_idx);
                Real birth = fil.value_by_sorted_id(simplex_idx);
                Real death = fil.infinity();

                result.add_point(dim, birth, death, simplex_idx, plus_inf, simplex_idx_us, plus_inf);
            } else {
                // finite point
                Int birth_idx = fil.index_in_filtration(low(col), dualize()), death_idx = simplex_idx;
				Int birth_idx_us = fil.get_id_by_sorted_id(birth_idx), death_idx_us = fil.get_id_by_sorted_id(death_idx);
                dim_type dim = fil.dim_by_sorted_id(birth_idx);
                Real birth = fil.value_by_sorted_id(birth_idx), death = fil.value_by_sorted_id(death_idx);

                bool include_point = include_all or (only_zero_persistence ? birth == death : birth != death);

                if (not include_point)
                    continue;

                if (dualize()) {
                    result.add_point(dim - 1, death, birth, death_idx, birth_idx, death_idx_us, birth_idx_us);
                } else {
                    result.add_point(dim, birth, death, birth_idx, death_idx, birth_idx_us, death_idx_us);
                }
            }

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
            if (col_idx % 100 == 0) {
                OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
            }
#endif

        }

        return result;
    }

    template<class Int>
    template<class Cell, class Real>
    Diagrams<Real> VRUDecomposition<Int>::diagram_general(const Filtration<Cell, Real>& fil,
            const typename Cell::UidSet& relative,
            bool include_all, bool include_inf_points, bool only_zero_persistence) const
    {
        if (not is_reduced)
            throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

        Diagrams<Real> result(fil.max_dim());

        std::unordered_set<Int> rows_with_lowest_one;

        if (include_all) {
            include_inf_points = true;
            only_zero_persistence = false;
        } else if (only_zero_persistence) {
            include_inf_points = false;
        }

        if (include_inf_points)
            for(size_t i = 0; i < r_data.size(); ++i) {

                // skip cells in relative
                auto column_cell = fil.get_cell(fil.index_in_filtration(i, dualize()));
                auto column_uid = column_cell.get_uid();
                if (relative.find(column_uid) != relative.end())
                    continue;

                if (!is_zero(&r_data[i]))
                    rows_with_lowest_one.insert(low(&r_data[i]));

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                if (i % 100 == 0) {
                    OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
                }
#endif

            }

        for(size_t col_idx = 0; col_idx < r_data.size(); ++col_idx) {

            auto col_cell = fil.get_cell(fil.index_in_filtration(col_idx, dualize()));
            auto col_uid = col_cell.get_uid();

            if (relative.find(col_uid) != relative.end()) {
                assert(is_zero(r_data[col_idx]));
                continue;
            }

            auto col = &r_data[col_idx];

            auto simplex_idx = fil.index_in_filtration(col_idx, dualize());
			auto simplex_idx_us = fil.get_id_by_sorted_id(simplex_idx);

            if (is_zero(col)) {
                if (not include_inf_points or rows_with_lowest_one.count(col_idx) != 0)
                    // we don't want infinite points or col_idx is a negative simplex
                    continue;

                // point at infinity
                dim_type dim = fil.dim_by_sorted_id(simplex_idx);
                Real birth = fil.value_by_sorted_id(simplex_idx);
                Real death = fil.infinity();

                result.add_point(dim, birth, death, simplex_idx, plus_inf, simplex_idx_us, plus_inf);
            } else {
                // finite point
                Int birth_idx = fil.index_in_filtration(low(col), dualize()), death_idx = simplex_idx;
				Int birth_idx_us = fil.get_id_by_sorted_id(birth_idx), death_idx_us = fil.get_id_by_sorted_id(death_idx);
                dim_type dim = fil.dim_by_sorted_id(birth_idx);
                Real birth = fil.value_by_sorted_id(birth_idx), death = fil.value_by_sorted_id(death_idx);

                bool include_point = include_all or (only_zero_persistence ? birth == death : birth != death);

                if (not include_point)
                    continue;

                if (dualize()) {
                    result.add_point(dim - 1, death, birth, death_idx, birth_idx, death_idx_us, birth_idx_us);
                } else {
                    result.add_point(dim, birth, death, birth_idx, death_idx, birth_idx_us, death_idx_us);
                }
            }

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
            if (col_idx % 100 == 0) {
                OINEUS_CHECK_FOR_PYTHON_INTERRUPT;
            }
#endif
        }

        return result;
    }

    template<class Int>
    template<class Cell, class Real>
    Diagrams<Real> VRUDecomposition<Int>::diagram(const Filtration<Cell, Real>& fil, bool include_inf_points) const
    {
        return diagram_general(fil, false, include_inf_points, false);
    }

    template<class Int>
    template<class Cell, class Real>
    Diagrams<Real> VRUDecomposition<Int>::diagram(const Filtration<Cell, Real>& fil, const typename Cell::UidSet& relative, bool include_inf_points) const
    {
        return diagram_general(fil, relative, false, include_inf_points, false);
    }

    template<class Int>
    template<class Cell, class Real>
    Diagrams<Real> VRUDecomposition<Int>::zero_persistence_diagram(const Filtration<Cell, Real>& fil) const
    {
        return diagram_general(fil, false, false, true);
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
