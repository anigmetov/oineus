#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

#include "interrupt.h"

#ifndef OINEUS_DISABLE_ICECREAM
#include <icecream/icecream.hpp>
#else
#ifndef IC
#define IC(...)
#endif
#ifndef IC0
#define IC0()
#endif
#endif

#include "common_defs.h"
#include "diagram.h"
#include "mem_reclamation.h"
#include "sparse_matrix.h"
#include "column_repr.h"
#include "timer.h"
#include "dcmp_stats.h"

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

        // Reusable per-thread working column. load_to_cache fills it in place
        // (clearing first), so dense representations (Full, BitTree) are sized
        // once here instead of being reallocated for every column.
        typename MatrixTraits::CachedColumn cached_reduced_col;
        MatrixTraits::reserve(cached_reduced_col, pivots.size());

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

                MatrixTraits::load_to_cache(orig_col, cached_reduced_col);

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
                            MatrixTraits::load_to_cache(orig_col, cached_reduced_col);
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
            // Worker context (std::thread): MUST NOT throw on interrupt
            // -- uncaught exceptions from std::thread call std::terminate.
            // Set-and-return; the orchestrator (reduce_parallel_*) checks
            // oineus::interrupted() after join() and throws there.
            if (my_chunk % 100 == 0 && oineus::interrupted())
                return;
            mm->quiescent();
        } while(true); // loop over chunks
        logger->debug("thread {}, EXIT reduction, stats = {}", thread_idx, stats);
    }

    template<class Int>
    size_t restore_elz_column_serial(std::vector<std::vector<Int>>& r_data,
            std::vector<std::vector<Int>>& v_data,
            std::vector<std::vector<Int>>* u_data_t,
            size_t current_col,
            bool v_only)
    {
        using MatrixTraits = SimpleSparseMatrixTraits<Int, 2>;

        size_t n_fixes = 0;
        size_t bottom_offset = 0;

        auto& current_v_col = v_data[current_col];
        auto& current_r_col = r_data[current_col];

        // Scan V from the bottom. If we undo an addition at row k, entries strictly below k
        // are unchanged (upper-triangular V), so `bottom_offset` remains valid.
        while (bottom_offset < current_v_col.size()) {
            const size_t v_idx = current_v_col.size() - 1 - bottom_offset;
            const Int added_col = current_v_col[v_idx];

            const bool is_current_col_death = not MatrixTraits::is_zero(current_r_col);
            const bool is_added_col_zero = MatrixTraits::is_zero(r_data[added_col]);

            const bool added_zero_column = (added_col < static_cast<Int>(current_col) && is_added_col_zero);
            const bool added_non_killing_column = (is_current_col_death && !is_added_col_zero &&
                    (MatrixTraits::low(&current_r_col) > MatrixTraits::low(&r_data[added_col])));

            if (added_zero_column || added_non_killing_column) {
                MatrixTraits::add_to_column(current_v_col, v_data[added_col]);
                if (not v_only) {
                    MatrixTraits::add_to_column(current_r_col, r_data[added_col]);
                    if (u_data_t) {
                        MatrixTraits::add_to_column((*u_data_t)[added_col], (*u_data_t)[current_col]);
                    }
                }
                ++n_fixes;
            } else {
                ++bottom_offset;
            }
        }

        return n_fixes;
    }

    template<class Int>
    bool restore_elz_column_parallel(typename SimpleRVMatrixTraits<Int, 2>::AMatrix& r_v_matrix, size_t current_col)
    {
        using SparseTraits = SimpleSparseMatrixTraits<Int, 2>;
        using RVTraits = SimpleRVMatrixTraits<Int, 2>;
        using RVColumn = typename RVTraits::Column;

        auto current_ptr = r_v_matrix[current_col].load(std::memory_order_relaxed);
        if (current_ptr == nullptr)
            return false;

        RVColumn local_col(*current_ptr);

        size_t bottom_offset = 0;
        bool changed = false;

        // Same bottom-up continuation logic as the serial version, but working on a local copy
        // of the current column before publishing a new pointer.
        while (bottom_offset < local_col.v_column.size()) {
            const size_t v_idx = local_col.v_column.size() - 1 - bottom_offset;
            const Int added_col = local_col.v_column[v_idx];

            auto added_ptr = r_v_matrix[added_col].load(std::memory_order_relaxed);
            if (added_ptr == nullptr)
                return changed;

            const bool is_current_col_death = not local_col.r_column.empty();
            const bool is_added_col_zero = added_ptr->r_column.empty();

            const bool added_zero_column = (added_col < static_cast<Int>(current_col) && is_added_col_zero);
            const bool added_non_killing_column = (is_current_col_death && !is_added_col_zero &&
                    (local_col.r_column.back() > added_ptr->r_column.back()));

            if (added_zero_column || added_non_killing_column) {
                SparseTraits::add_to_column(local_col.v_column, added_ptr->v_column);
                SparseTraits::add_to_column(local_col.r_column, added_ptr->r_column);
                changed = true;
            } else {
                ++bottom_offset;
            }
        }

        if (changed) {
            auto* new_col = new RVColumn(std::move(local_col.r_column), std::move(local_col.v_column));
            r_v_matrix[current_col].store(new_col, std::memory_order_relaxed);
        }

        return changed;
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
        // True iff R, V are known to be in ELZ form. Maintained by the
        // reduction drivers and by restore_elz; consulted by
        // compute_partial_u_rows which requires V to be ELZ. A full
        // re-check via is_elz() walks the whole matrix and is too
        // expensive for the gradient-loop hot path.
        std::map<dim_type, bool> is_elz_in_dim_;

        // in parallel versions we use atomic pivots, in serial - normal
        std::vector<Int> _pivots;

        // Row-incidence indices (row -> sorted columns containing it) for R, V,
        // D, used to localize dynamic updates (vineyards / moves / warm starts)
        // to the changed support instead of scanning the whole matrix. Built
        // on demand by make_dynamic() (reusing the parallel col->row transpose)
        // and maintained incrementally by the manipulation methods. A static
        // decomposition never builds them. Invalidated by reduce() and any
        // resize.
        MatrixData ri_r_;
        MatrixData ri_v_;
        MatrixData ri_d_;
        bool is_dynamic_ {false};

        std::vector<Int> dim_first;
        std::vector<Int> dim_last;

        // for cohomology: reverse dims,
        // corresponds to dimension of coboundary matrix
        std::vector<Int> _dim_first;
        std::vector<Int> _dim_last;

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
                dim_first(fil.dims_first()),
                dim_last(fil.dims_last()),
                _dim_first(fil.dims_first()),
                _dim_last(fil.dims_last()),
                n_rows(d_data.size())
        {
            if (dualize_) {
                std::reverse(_dim_first.begin(), _dim_first.end());
                std::reverse(_dim_last.begin(), _dim_last.end());
                std::vector<Int> new_dim_first, new_dim_last;
                for(size_t i = 0; i < dim_first.size(); ++i) {
                    size_t cnt = _dim_last[i] - _dim_first[i];
                    if (i == 0) {
                        new_dim_first.push_back(0);
                        new_dim_last.push_back(cnt);
                    } else {
                        new_dim_first.push_back(new_dim_last.back() + 1);
                        new_dim_last.push_back(new_dim_first.back() + cnt);
                    }
                }
                _dim_first = new_dim_first;
                _dim_last = new_dim_last;
            }

            for(dim_type _dim = 0; _dim < dim_first.size(); ++_dim) {
                is_elz_in_dim_[_dim] = false;
            }
        }

        // Construct from a pre-built boundary matrix plus explicit dim
        // arrays from the source filtration. When dualize=true, the
        // antitranspose of bdry is used (this is what the existing
        // filtration ctor does internally when called with _dualize=true).
        // TopologyOptimizer uses this ctor to share one boundary matrix
        // across both hom and coh decompositions; building this ctor lets
        // us skip the redundant filtration walk that the templated ctor
        // would otherwise perform.
        VRUDecomposition(const MatrixData& bdry,
                         std::vector<Int> dim_first_,
                         std::vector<Int> dim_last_,
                         bool dualize,
                         int n_threads = 1)
                :
                d_data(dualize ? antitranspose(bdry, bdry.size()) : bdry),
                r_data(d_data),
                dualize_(dualize),
                dim_first(std::move(dim_first_)),
                dim_last(std::move(dim_last_)),
                _dim_first(dim_first),
                _dim_last(dim_last),
                n_rows(d_data.size())
        {
            if (dualize_) {
                std::reverse(_dim_first.begin(), _dim_first.end());
                std::reverse(_dim_last.begin(), _dim_last.end());
                std::vector<Int> new_dim_first, new_dim_last;
                for(size_t i = 0; i < dim_first.size(); ++i) {
                    size_t cnt = _dim_last[i] - _dim_first[i];
                    if (i == 0) {
                        new_dim_first.push_back(0);
                        new_dim_last.push_back(cnt);
                    } else {
                        new_dim_first.push_back(new_dim_last.back() + 1);
                        new_dim_last.push_back(new_dim_first.back() + cnt);
                    }
                }
                _dim_first = std::move(new_dim_first);
                _dim_last = std::move(new_dim_last);
            }
            for(dim_type _dim = 0; _dim < static_cast<dim_type>(dim_first.size()); ++_dim) {
                is_elz_in_dim_[_dim] = false;
            }
            (void) n_threads;
        }

        VRUDecomposition(const MatrixData& d, size_t n_rows = std::numeric_limits<decltype(n_rows)>::max(), bool dualize=false, bool skip_check=false)
                :
                d_data(d),
                r_data(d),
                dualize_(dualize),
                // TODO: think about dimensions here
                dim_first(std::vector<Int>({0})),
                dim_last(std::vector<Int>({static_cast<Int>(d.size() - 1)})),
                _dim_first(std::vector<Int>({0})),
                _dim_last(std::vector<Int>({static_cast<Int>(d.size() - 1)})),
                n_rows(n_rows == std::numeric_limits<decltype(n_rows)>::max() ? d_data.size() : n_rows)
        {
            for(dim_type _dim = 0; _dim < dim_first.size(); ++_dim) {
                is_elz_in_dim_[_dim] = false;
            }

            if (!skip_check) {
                for(auto&& col : d) {
                    for(auto&& e: col) {
                        if (static_cast<decltype(n_rows)>(e) >= n_rows) {
                            throw std::runtime_error("Row index out of range, specify a bigger value for n_rows");
                        }
                    }
                }
            }
        }

        [[nodiscard]] size_t size() const { return r_data.size(); }

        bool dualize() const { return dualize_; }

        bool operator==(const VRUDecomposition& other) const
        {
            return d_data == other.d_data
                && r_data == other.r_data
                && v_data == other.v_data
                && u_data_t == other.u_data_t
                && is_reduced == other.is_reduced
                && dualize_ == other.dualize_
                && _pivots == other._pivots
                && dim_first == other.dim_first
                && dim_last == other.dim_last
                && _dim_first == other._dim_first
                && _dim_last == other._dim_last
                && n_rows == other.n_rows;
        }

        bool operator!=(const VRUDecomposition& other) const
        {
            return !(*this == other);
        }

        void set_is_elz_flag(dim_type _dim, bool new_value);

        void reduce(Params& params);

        // Public dispatchers: select the working-column representation from
        // params.col_repr and forward to the templated *_impl below.
        void reduce_serial(Params& params);
        void reduce_parallel_r_only(Params& params);
        void reduce_parallel_rv(Params& params);

        // Templated reduction kernels, one instantiation per working-column type
        // (WorkCol = SetColumn<Int>, HeapColumn<Int>, FullColumn<Int>, BitTreeColumn<Int>).
        template<class WorkCol> void reduce_serial_impl(Params& params);
        template<class WorkCol> void reduce_parallel_r_only_impl(Params& params);
        template<class WorkCol> void reduce_parallel_rv_impl(Params& params);

        // Copy params.timings into the back-compat scalar timing fields.
        static void sync_elapsed_from_timings_(Params& params);

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

        int is_column_elz(size_t col_idx) const;

        bool is_elz(int n_threads=8) const;
        size_t n_elz_violators(int n_threads) const;
        size_t n_elz_violators_in_dim(dim_type dim, int n_threads) const;
        // std::vector<int8_t> mark_elz_violators_in_dim(dim_type dim, int n_threads) const;

        void restore_elz(dim_type dim, bool v_only, bool verbose, int n_threads);

        size_t n_dims() const { return dim_first.size(); }

        // convert geometric dim into index in _dim_first/last vectors (same
        // for homology, reversed for cohomology)
        // internally use _dim to index, accept dim as parameter
        size_t _dim_from_dim(dim_type dim) const { return dualize() ? n_dims() - dim - 1 : dim; }

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

        // ===== decomposition manipulation (vineyards / moves / warm starts) =====
        // Update a reduced R = DV decomposition when the filtration changes,
        // instead of recomputing from scratch. All require
        // is_reduced && has_matrix_v() && not dualize() (homology only).
        // d_data is kept consistent so sanity_check() stays a valid oracle;
        // u_data_t is invalidated (cleared). The optional stats pointer
        // accumulates per-phase wall-clock and column-operation counts.
        //
        // Permutations are given as `new_to_old` vectors: new_to_old[k] is the
        // old matrix index now occupying position k.

        // Opt into dynamic mode: build the row-incidence indices (in parallel)
        // so subsequent manipulations cost O(changed support) instead of
        // O(whole matrix). Optional -- the manipulation methods build the
        // indices on first use if absent. reduce() and resizes drop them.
        void make_dynamic(int n_threads = 1);
        bool is_dynamic() const { return is_dynamic_; }

        // Vineyards (Cohen-Steiner; Piekenbrock-Perea Alg 6): transpose the
        // adjacent filtration positions i and i+1. The two cells must be
        // transposable (neither a face of the other); this holds whenever
        // they have equal dimension, which is the only case transpose_to /
        // apply_move_schedule ever generate.
        void transpose(size_t i, DecompositionManipStats* stats = nullptr);

        // Realize target order `new_to_old` as a sequence of adjacent
        // transpositions (per-dimension). Returns the number performed.
        size_t transpose_to(const std::vector<size_t>& new_to_old, DecompositionManipStats* stats = nullptr);

        // Moves (Busaryev; Piekenbrock-Perea 2.4): move the cell at position i
        // to position j (R = DV is invalid mid-call, valid on return).
        void move(size_t i, size_t j, DecompositionManipStats* stats = nullptr);
        void move_right(size_t i, size_t j, DecompositionManipStats* stats = nullptr); // i < j
        void move_left(size_t i, size_t j, DecompositionManipStats* stats = nullptr);  // i > j

        // Move schedule (Piekenbrock-Perea Alg 4/5): realize `new_to_old` as a
        // minimal-size set of moves via a per-dimension LIS. Returns #moves.
        size_t apply_move_schedule(const std::vector<size_t>& new_to_old, DecompositionManipStats* stats = nullptr);

        // Luo-Nelson Alg 2: warm-start update under a pure reorder of a
        // fixed-size filtration to target order `new_to_old`. Conjugates the
        // existing (reduced) R, V, D by the reorder permutation and re-reduces
        // R, reusing the old factorization. Leaves R reduced with the correct
        // pairing and D V == R; V is a valid full-rank change of basis but is
        // NOT re-triangularized into the new order (that extra Luo-Nelson pass
        // does not affect the diagram and is deferred), so use
        // is_reduced_consistent() rather than sanity_check() as the oracle.
        void update_with_permutation(const std::vector<size_t>& new_to_old,
                                     DecompositionManipStats* stats = nullptr);

        // Luo-Nelson Alg 3: warm-start update with cell insertion / deletion
        // plus reorder; resizes the decomposition to new_boundary.size().
        //   new_to_old[k] = old index of the cell now at new position k, or -1
        //                   if the cell at position k is freshly inserted.
        //   new_boundary  = full boundary matrix of the new filtration
        //                   (n_new columns, new sorted order).
        // Deletions must be coface-closed (no surviving cell may have a deleted
        // face), which any valid filtration edit satisfies. Reuses the survivor
        // factorization: reorders survivors to the front via Alg 2, truncates
        // the deleted tail, then re-reduces only R over the new (warm) basis.
        // new_dim_first/new_dim_last carry the dimension-block layout of the new
        // filtration so the decomposition stays consistent with Oineus's
        // dimension-blocked model.
        void update_with_edits(const std::vector<long long>& new_to_old,
                               const MatrixData& new_boundary,
                               const std::vector<Int>& new_dim_first,
                               const std::vector<Int>& new_dim_last,
                               DecompositionManipStats* stats = nullptr);

        // --- manipulation helpers ---
        // Throw std::runtime_error if a manipulation cannot be applied
        // (not reduced, no V, or cohomology).
        void check_manip_preconditions_(const char* who) const;

        // Reduce red[col] against `pivots` (pivots[low] = owning column index,
        // -1 if none), mirroring each column-add into mir; when a fresh pivot
        // appears, set pivots[low(red[col])] = col. Mirrors the inner loop of
        // reduce_serial_impl but operates on the at-rest sorted columns and
        // resumes from a given pivot state. n_add_red/n_add_mir accumulate
        // the column-op counts.
        void reduce_column_with_pivots_(MatrixData& red, MatrixData& mir, size_t col,
                                        std::vector<Int>& pivots,
                                        long long& n_add_red, long long& n_add_mir);

        static std::vector<size_t> invert_perm(const std::vector<size_t>& p);

        // Throw if p is not a permutation of [0, n): wrong size, out of range,
        // or duplicate entries.
        static void validate_permutation_(const std::vector<size_t>& p, size_t n, const char* who);

        // Boolean mask of one longest strictly-increasing subsequence of `a`.
        static std::vector<char> lis_mask(const std::vector<size_t>& a);

        // Swap rows i and i+1 in a column-stored matrix. Because i and i+1 are
        // adjacent integers, relabeling i<->i+1 inside a sorted column keeps it
        // sorted, so no re-sort is needed. O(total non-zeros).
        static void swap_adjacent_rows_(MatrixData& m, Int i);

        // --- dynamic-mode (row-index) helpers ---
        void ensure_dynamic_(int n_threads = 1);
        void invalidate_dynamic_();
        // a row's incidence list is a sorted std::vector<Int> of column indices
        static bool ri_contains_(const IntSparseColumn& lst, Int c);
        static void ri_insert_(IntSparseColumn& lst, Int c);
        static void ri_remove_(IntSparseColumn& lst, Int c);
        // Indexed column ops that keep the row index `ri` (for matrix `m`) in
        // sync. col_add: m[dst] += src_col. col_swap: swap m[a], m[b].
        // row_swap_adjacent: swap rows i, i+1 of m, touching only columns the
        // index says contain those rows.
        static void col_add_indexed_(MatrixData& m, MatrixData& ri, size_t dst, const IntSparseColumn& src_col);
        static void col_swap_indexed_(MatrixData& m, MatrixData& ri, size_t a, size_t b);
        static void row_swap_adjacent_indexed_(MatrixData& m, MatrixData& ri, Int i);

        // Relabel row entries of every column by `map` (r -> map[r]); re-sorts
        // each column (an arbitrary relabel breaks sortedness). Columns are not
        // reordered. (= left-multiply by the permutation matrix.)
        static void relabel_rows_(MatrixData& m, const std::vector<size_t>& map);

        // Relaxed validity check for warm-start updates that do not
        // re-triangularize V: R is reduced and D V == R (over Z_2). Does NOT
        // require V upper-triangular (unlike sanity_check).
        bool is_reduced_consistent() const;

        // Total non-zeros (sum of column sizes) of a column-stored matrix.
        static size_t matrix_nnz_(const MatrixData& m);

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

        IntSparseColumn compute_u_column(size_t col_idx) const;
        void compute_u_from_v(dim_type dim, size_t n_threads=1, bool verbose=false);

        IntSparseColumn compute_u_column_1(size_t col_idx) const;
        void compute_u_from_v_1(dim_type dim, size_t n_threads=1, bool verbose=false);

        // Row-form U primitives. Solves (row r of U) V = e_r^T in
        // residual style against V^T (lower unit-triangular, forward
        // substitution). Each row solve is independent and writes its
        // row directly into u_data_t -- no col->row stage is needed.
        // Caller must build vt_data once via
        // MatrixTraits::col_to_row_format_parallel restricted to the
        // dim of interest, and pass it in.
        //
        // cmp_op direction depends on dualize x walker direction:
        //   hom (dualize=false), increase_death walker:
        //     cmp_op(piv_value, bound) = (piv_value > bound)  ["above"]
        //   coh (dualize=true), decrease_birth walker:
        //     cmp_op(piv_value, bound) = (piv_value < bound)  ["below"]
        // Negate flips both directions; not yet supported.
        template<typename Real, typename ValueAt, typename CmpOp>
        IntSparseColumn compute_u_row_bounded(size_t row_idx,
                                              const MatrixData& vt_data,
                                              Real value_bound,
                                              ValueAt&& value_at,
                                              CmpOp&& cmp_op) const;

        // Parallel partial-rows driver. Builds vt_data internally for
        // `dim`, then runs n_threads row solves on the rows list.
        // u_data_t is sized to v_data.size() if not already; only the
        // rows[i] slots are written. cmp_op is shared across rows
        // (same direction; see table above).
        template<typename Real, typename ValueAt, typename CmpOp>
        void compute_partial_u_rows(const std::vector<size_t>& rows,
                                    const std::vector<Real>& bounds,
                                    dim_type dim,
                                    ValueAt&& value_at,
                                    CmpOp&& cmp_op,
                                    size_t n_threads = 1,
                                    bool verbose = false);

        // Full-rows pass: same machinery but with rows = entire dim
        // and cmp_op = never_stop. Equivalent to compute_u_from_v_1
        // for the dim, but produces u_data_t directly without a
        // col->row conversion.
        template<typename Real, typename ValueAt>
        void compute_full_u_rows(dim_type dim,
                                 ValueAt&& value_at,
                                 size_t n_threads = 1,
                                 bool verbose = false);

        size_t range_start_(dim_type _dim) const;
        size_t range_end_(dim_type _dim) const;
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
            if (rr.col(i) != dv.col(i)) {
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
    int VRUDecomposition<Int>::is_column_elz(size_t col_idx) const
    {
        assert(has_matrix_v());
        bool is_death = not is_R_column_zero(col_idx);
        for(auto row_idx : v_data[col_idx]) {
            if (row_idx == static_cast<Int>(col_idx))
                continue;
            // we added column that was eventually zeroed
            if (is_R_column_zero(row_idx))
                return 0;
            // in the ELZ reduction, this column would not be added: it's low is higher than ours,
            // so it does not kill anything
            if (is_death and r_data[col_idx].back() > r_data[row_idx].back())
                return 0;
        }
        return 1;
    }

    template<class Int>
    bool VRUDecomposition<Int>::is_elz(int n_threads) const
    {
        if (not has_matrix_v()) {
            throw std::runtime_error("VRUDecomposition: cannot check ELZ without V matrix");
        }

        size_t n_cols = r_data.size();

        // Don't use multithreading if too few columns
        if (n_threads == 1 or n_cols < static_cast<size_t>(8 * n_threads)) {
            // Serial version
            for (size_t col_idx = 0; col_idx < n_cols; ++col_idx) {
                if (not is_column_elz(col_idx)) {
                    return false;
                }
            }
            return true;
        }

        // Parallel version
        std::atomic<bool> found_violator{false};

        tf::Executor executor(n_threads);
        tf::Taskflow taskflow;

        taskflow.for_each_index(size_t(0), n_cols, size_t(1),
            [this, &found_violator](size_t col_idx) {
                // Check flag before processing each column
                if (found_violator.load(std::memory_order_relaxed)) {
                    return;
                }

                if (not is_column_elz(col_idx)) {
                    found_violator.store(true, std::memory_order_relaxed);
                }
            });

        executor.run(taskflow).wait();

        return !found_violator.load();
    }

    template<class Int>
    size_t VRUDecomposition<Int>::n_elz_violators(int n_threads) const
    {
        size_t result = 0;
        for(dim_type d = 0; d < _dim_first.size(); ++ d) {
            result += n_elz_violators_in_dim(d, n_threads);
        }
        return result;
    }


    template<class Int>
    size_t VRUDecomposition<Int>::n_elz_violators_in_dim(dim_type dim, int n_threads) const
    {
        if (not has_matrix_v()) {
            throw std::runtime_error("VRUDecomposition: cannot check ELZ without V matrix");
        }

        auto _dim = _dim_from_dim(dim);

        size_t start_idx = _dim_first.at(_dim);
        size_t end_idx = _dim_last.at(_dim) + 1;

        size_t n_cols_to_check = end_idx - start_idx;

        // Don't use multithreading if too few columns
        if (n_threads == 1 or n_cols_to_check < static_cast<size_t>(8 * n_threads)) {
            // Serial version
            size_t count = 0;
            for (size_t col_idx = start_idx; col_idx < end_idx; ++col_idx) {
                if (!is_column_elz(col_idx)) {
                    count++;
                }
            }
            return count;
        }

        // Parallel version - each thread maintains its own counter
        std::vector<size_t> thread_counts(n_threads, 0);

        tf::Executor executor(n_threads);
        tf::Taskflow taskflow;

        // Calculate chunk size for each thread
        size_t chunk_size = (n_cols_to_check + n_threads - 1) / n_threads;

        for (int tid = 0; tid < n_threads; ++tid) {
            taskflow.emplace([this, tid, start_idx, end_idx, chunk_size, n_cols_to_check, &thread_counts]() {
                size_t thread_start = start_idx + tid * chunk_size;
                size_t thread_end = std::min(thread_start + chunk_size, end_idx);

                size_t local_count = 0;
                for (size_t col_idx = thread_start; col_idx < thread_end; ++col_idx) {
                    if (!is_column_elz(col_idx)) {
                        local_count++;
                    }
                }
                thread_counts[tid] = local_count;
            });
        }

        executor.run(taskflow).wait();

        // Sum up counts from all threads
        size_t total_count = 0;
        for (size_t count : thread_counts) {
            total_count += count;
        }

        return total_count;
    }

    // return the beginning of interesting range
    // if dim > top dimensions, use all dimensions except 0 for homology
    template<class Int>
    size_t VRUDecomposition<Int>::range_start_(dim_type _dim) const
    {
        if (_dim < n_dims() and _dim >= 0)
            return _dim_first[_dim];
        // in cohomology, coboundary of vertices is interesting
        if (dualize() or n_dims() == 0)
            return 0;
        // in homology, boundary of vertices is empty, we should start from dim 1
        return _dim_first[1];
    }

    // return the end of interesting range
    // if dim > top dimensions, use all dimensions except top_dim for cohomology
    template<class Int>
    size_t VRUDecomposition<Int>::range_end_(dim_type _dim) const
    {
        if (_dim < n_dims() and _dim >= 0)
            return _dim_last[_dim] + 1;
        // all dims
        if (dualize()) {
            // for cohomology: 0..dim-1
            if (n_dims() > 2) {
                return _dim_last[n_dims() - 2] + 1;
            } else {
                return _dim_last.back() + 1;
            }
        }
        // for homology: 1...dim
        return _dim_last.back() + 1;
    }

    template<class Int>
    void VRUDecomposition<Int>::set_is_elz_flag(dim_type _dim, bool new_value)
    {
        if (_dim != k_all_dims) {
            is_elz_in_dim_[_dim] = new_value;
        } else {
            for(dim_type d = 0; d < _dim_first.size(); ++d) {
                is_elz_in_dim_[d] = new_value;
            }
        }
    }

    template<class Int>
    void VRUDecomposition<Int>::restore_elz(dim_type dim, bool v_only, bool verbose, int n_threads)
    {
        if (not has_matrix_v()) {
            throw std::runtime_error("VRUDecomposition: cannot restore ELZ without V matrix");
        }

        size_t n_violators = 0;
        const auto _dim = _dim_from_dim(dim);
        const size_t range_start_idx = range_start_(_dim);
        const size_t range_end_idx = range_end_(_dim);

        if (verbose) { IC(dim, range_start_idx, range_end_idx); }

        Timer timer;

        for (size_t current_col = range_start_idx; current_col < range_end_idx; ++current_col) {
            n_violators += restore_elz_column_serial(r_data, v_data, has_matrix_u() ? &u_data_t : nullptr, current_col, v_only);

            if (current_col % 100 == 0 && oineus::interrupted())
                throw oineus::interrupted_exception{};
        }

        set_is_elz_flag(_dim, true);

        if (verbose) { IC(n_violators, size(), timer.elapsed()); }
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;

        invalidate_dynamic_();   // R, V are rebuilt; any row index is now stale
        params.timings.reset();

        if (d_data.empty()) {
            is_reduced = true;
            sync_elapsed_from_timings_(params);
            return;
        }

        if (params.n_threads > 1 and params.compute_u)
            throw std::runtime_error("Cannot compute U matrix in parallel");

        // Serial + no clearing already produces ELZ, so restore_elz is ignored there.
        const bool serial_without_clearing = (params.n_threads == 1 && !params.clearing_opt);
        if (not params.dims_to_restore_elz.empty() and not params.compute_v and not serial_without_clearing)
            throw std::runtime_error("Cannot restore ELZ during reduction without V matrix");

        if (params.n_threads == 1)
            reduce_serial(params);
        else if (params.compute_v)
            reduce_parallel_rv(params);
        else
            reduce_parallel_r_only(params);

        // Derive the back-compat scalar timers from the per-phase breakdown:
        // elapsed is now the full, path-comparable reduction time.
        sync_elapsed_from_timings_(params);
    }

    // Mirror params.timings into the historical scalar timing fields so existing
    // callers (and the pickle layout) keep working. elapsed becomes the total.
    template<class Int>
    void VRUDecomposition<Int>::sync_elapsed_from_timings_(Params& params)
    {
        params.elapsed             = params.timings.reduction_total();
        params.elapsed_restore_elz = params.timings.restore_elz;
        params.elapsed_copy_back   = params.timings.copy_back;
        params.elapsed_copy_pivots = params.timings.copy_pivots;
    }

    // ---- working-column dispatchers: pick WorkCol from params.col_repr ----

    template<class Int>
    void VRUDecomposition<Int>::reduce_serial(Params& params)
    {
        switch (params.col_repr) {
            case ColumnRepr::Set:     reduce_serial_impl<SetColumn<Int>>(params); break;
            case ColumnRepr::Heap:    reduce_serial_impl<HeapColumn<Int>>(params); break;
            case ColumnRepr::Full:    reduce_serial_impl<FullColumn<Int>>(params); break;
            case ColumnRepr::BitTree: reduce_serial_impl<BitTreeColumn<Int>>(params); break;
        }
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce_parallel_r_only(Params& params)
    {
        switch (params.col_repr) {
            case ColumnRepr::Set:     reduce_parallel_r_only_impl<SetColumn<Int>>(params); break;
            case ColumnRepr::Heap:    reduce_parallel_r_only_impl<HeapColumn<Int>>(params); break;
            case ColumnRepr::Full:    reduce_parallel_r_only_impl<FullColumn<Int>>(params); break;
            case ColumnRepr::BitTree: reduce_parallel_r_only_impl<BitTreeColumn<Int>>(params); break;
        }
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce_parallel_rv(Params& params)
    {
        switch (params.col_repr) {
            case ColumnRepr::Set:     reduce_parallel_rv_impl<SetColumn<Int>>(params); break;
            case ColumnRepr::Heap:    reduce_parallel_rv_impl<HeapColumn<Int>>(params); break;
            case ColumnRepr::Full:    reduce_parallel_rv_impl<FullColumn<Int>>(params); break;
            case ColumnRepr::BitTree: reduce_parallel_rv_impl<BitTreeColumn<Int>>(params); break;
        }
    }

    template<class Int>
    template<class WorkCol>
    void VRUDecomposition<Int>::reduce_serial_impl(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;

        // If clearing is off, serial reduction is already ELZ and restore_elz is ignored.
        if (not params.dims_to_restore_elz.empty() and params.clearing_opt and not params.compute_v) {
            throw std::runtime_error("Cannot restore ELZ during serial reduction without V matrix");
        }

        Timer timer_reduction;

        using MatrixTraits = GenericSparseMatrixTraits<Int, WorkCol>;

        ThreadStats stats {0};
        int n_cleared = 0;

        std::unordered_set<Int> cleared_cols;

        if (params.compute_v)
            v_data = MatrixTraits::eye(d_data.size());

        if (params.compute_u)
            u_data_t = MatrixTraits::eye(d_data.size());

        // D matrix empty, nothing to do
        if (d_data.empty()) {
            is_reduced = true;
            return;
        }

        _pivots = std::vector<Int>(n_rows, -1);

        // homology: go from top dimension to 0, to make clearing possible
        // cohomology: the opposite
        // NB: _dim_first, _dim_last were reversed in ctor for cohomology!
        // Reusable working columns, filled in place per column so dense
        // representations are sized once rather than reallocated per column.
        typename MatrixTraits::CachedColumn cached_r_col;
        typename MatrixTraits::CachedColumn cached_v_col;
        MatrixTraits::reserve(cached_r_col, n_rows);
        if (params.compute_v)
            MatrixTraits::reserve(cached_v_col, n_rows);

        for(int dim = _dim_first.size() - 1; dim >= 0; --dim) {
            for(Int i = _dim_first[dim]; i <= _dim_last[dim]; ++i) {
                if (params.clearing_opt and not is_zero(r_data[i])) {
                    // simplex i is pivot -> i is positive -> its column is 0
                    if (_pivots[i] >= 0) {
                        r_data[i].clear();
                        n_cleared++;
                        // U. Bauer's trick to get a valid V column for cleared columns
                        if (params.compute_v) {
                            v_data[i] = r_data[_pivots[i]];
                        }
                        continue;
                    }
                }

                MatrixTraits::load_to_cache(r_data[i], cached_r_col);

                if (params.compute_v) {
                    MatrixTraits::load_to_cache(v_data[i], cached_v_col);
                }

                while(not MatrixTraits::is_zero(cached_r_col)) {

                    Int& pivot = _pivots[MatrixTraits::low(cached_r_col)];

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

                    if (i % 1000 == 0 && oineus::interrupted())
                        throw oineus::interrupted_exception{};
                } // reduction loop
                MatrixTraits::load_from_cache(cached_r_col, r_data[i]);
                if (params.compute_v)
                    MatrixTraits::load_from_cache(cached_v_col, v_data[i]);
            } // loop over columns in fixed dimension
        } // loop over dimensions

        // Serial reduces in place: no prepare / copy_back / copy_pivots phases.
        params.timings.reduce = timer_reduction.elapsed();

        // Serial reduction with clearing off produces ELZ by construction.
        if (not params.clearing_opt) {
            set_is_elz_flag(k_all_dims, true);
        }

        if (params.dims_to_restore_elz.size() > 0 and params.clearing_opt) {
            Timer timer_restore;
            for(auto dim : params.dims_to_restore_elz) {
                if (dim >= dim_first.size())
                    continue;
                restore_elz(dim, false, params.verbose, 1);
            }
            params.timings.restore_elz = timer_restore.elapsed();
        }

        if (params.print_time or params.verbose) {
            std::cerr << "reduce_serial, matrix_size = " << r_data.size()
                      << ", clearing_opt = " << params.clearing_opt
                      << ", n_cleared = " << n_cleared
                      << ", reduction elapsed: " << params.timings.reduce
                      << ", restore_elz elapsed: " << params.timings.restore_elz
                      << ", total elapsed: " << params.timings.reduction_total()
                      << std::endl;
        }

        is_reduced = true;
    }

    // ===================================================================
    // Decomposition manipulation: shared helpers
    // ===================================================================

    template<class Int>
    void VRUDecomposition<Int>::check_manip_preconditions_(const char* who) const
    {
        // Homology and cohomology are both fine: the manipulation methods are
        // generic R = D V maintenance in MATRIX-index space. For cohomology
        // (dualize), the matrix stores the antitransposed boundary in reversed
        // filtration order, the dimension blocks are _dim_first/_dim_last
        // (matrix space), and the caller must give the permutation in matrix
        // space (= reversal-conjugate of a filtration reorder). diagram() maps
        // matrix indices back via index_in_filtration(., dualize()).
        if (not is_reduced)
            throw std::runtime_error(std::string(who) + ": decomposition must be reduced first");
        if (not has_matrix_v())
            throw std::runtime_error(std::string(who) + ": V matrix is required (reduce with compute_v = true)");
    }

    template<class Int>
    std::vector<size_t> VRUDecomposition<Int>::invert_perm(const std::vector<size_t>& p)
    {
        std::vector<size_t> inv(p.size(), p.size());
        for(size_t i = 0; i < p.size(); ++i) {
            if (p[i] >= p.size() or inv[p[i]] != p.size())
                throw std::runtime_error("invert_perm: argument is not a permutation");
            inv[p[i]] = i;
        }
        return inv;
    }

    template<class Int>
    void VRUDecomposition<Int>::validate_permutation_(const std::vector<size_t>& p, size_t n, const char* who)
    {
        if (p.size() != n)
            throw std::runtime_error(std::string(who) + ": permutation size mismatch");
        std::vector<char> seen(n, 0);
        for(size_t e : p) {
            if (e >= n)
                throw std::runtime_error(std::string(who) + ": permutation entry out of range");
            if (seen[e])
                throw std::runtime_error(std::string(who) + ": permutation has duplicate entries");
            seen[e] = 1;
        }
    }

    template<class Int>
    std::vector<char> VRUDecomposition<Int>::lis_mask(const std::vector<size_t>& a)
    {
        const size_t n = a.size();
        std::vector<char> in_lis(n, 0);
        if (n == 0)
            return in_lis;
        std::vector<size_t> tails;          // tails[k] = index into a of the smallest tail of an increasing subseq of length k+1
        std::vector<size_t> prev(n, n);     // predecessor index in the reconstructed subsequence
        for(size_t idx = 0; idx < n; ++idx) {
            // strictly increasing: first tail whose value is >= a[idx]
            size_t lo = 0, hi = tails.size();
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (a[tails[mid]] < a[idx]) lo = mid + 1;
                else hi = mid;
            }
            if (lo > 0)
                prev[idx] = tails[lo - 1];
            if (lo == tails.size())
                tails.push_back(idx);
            else
                tails[lo] = idx;
        }
        size_t k = tails.back();
        while (k != n) {
            in_lis[k] = 1;
            k = prev[k];
        }
        return in_lis;
    }

    template<class Int>
    void VRUDecomposition<Int>::reduce_column_with_pivots_(
            MatrixData& red, MatrixData& mir, size_t col,
            std::vector<Int>& pivots, long long& n_add_red, long long& n_add_mir)
    {
        using ST = SimpleSparseMatrixTraits<Int, 2>;
        // Standard left-to-right reduction of a single column against the
        // current pivot state. When col shares its low with an earlier pivot
        // column, add that column (and its mirror) and repeat; the low
        // strictly decreases each step, so this terminates.
        while(not red[col].empty()) {
            Int lo = red[col].back();                 // low = largest row index (columns are sorted)
            Int piv = pivots[lo];
            if (piv == -1) {
                pivots[lo] = static_cast<Int>(col);   // fresh pivot for this row
                break;
            }
            if (static_cast<size_t>(piv) == col)
                break;                                // already owns this low
            ST::add_to_column(red[col], red[piv]);
            ++n_add_red;
            ST::add_to_column(mir[col], mir[piv]);
            ++n_add_mir;
        }
    }

    template<class Int>
    void VRUDecomposition<Int>::make_dynamic(int n_threads)
    {
        using ST = SimpleSparseMatrixTraits<Int, 2>;
        auto build = [&](const MatrixData& m) -> MatrixData {
            MatrixData ri = ST::col_to_row_format_parallel(m, n_threads, 0, m.size(), static_cast<Int>(n_rows));
            ri.resize(n_rows);          // one list per row, including trailing empty rows
            for(auto& lst : ri)
                std::sort(lst.begin(), lst.end());
            return ri;
        };
        ri_r_ = build(r_data);
        ri_v_ = build(v_data);
        ri_d_ = build(d_data);
        is_dynamic_ = true;
    }

    template<class Int>
    void VRUDecomposition<Int>::ensure_dynamic_(int n_threads)
    {
        if (not is_dynamic_)
            make_dynamic(n_threads);
    }

    template<class Int>
    void VRUDecomposition<Int>::invalidate_dynamic_()
    {
        is_dynamic_ = false;
        ri_r_.clear();
        ri_v_.clear();
        ri_d_.clear();
    }

    template<class Int>
    bool VRUDecomposition<Int>::ri_contains_(const IntSparseColumn& lst, Int c)
    {
        return std::binary_search(lst.begin(), lst.end(), c);
    }

    template<class Int>
    void VRUDecomposition<Int>::ri_insert_(IntSparseColumn& lst, Int c)
    {
        auto it = std::lower_bound(lst.begin(), lst.end(), c);
        if (it == lst.end() or *it != c)
            lst.insert(it, c);
    }

    template<class Int>
    void VRUDecomposition<Int>::ri_remove_(IntSparseColumn& lst, Int c)
    {
        auto it = std::lower_bound(lst.begin(), lst.end(), c);
        if (it != lst.end() and *it == c)
            lst.erase(it);
    }

    template<class Int>
    void VRUDecomposition<Int>::col_add_indexed_(MatrixData& m, MatrixData& ri, size_t dst, const IntSparseColumn& src_col)
    {
        using ST = SimpleSparseMatrixTraits<Int, 2>;
        const Int cdst = static_cast<Int>(dst);
        // each row r of src_col flips membership in m[dst], so toggle dst in ri[r]
        for(Int r : src_col) {
            auto& lst = ri[static_cast<size_t>(r)];
            auto it = std::lower_bound(lst.begin(), lst.end(), cdst);
            if (it != lst.end() and *it == cdst)
                lst.erase(it);
            else
                lst.insert(it, cdst);
        }
        ST::add_to_column(m[dst], src_col);
    }

    template<class Int>
    void VRUDecomposition<Int>::col_swap_indexed_(MatrixData& m, MatrixData& ri, size_t a, size_t b)
    {
        const Int ca = static_cast<Int>(a), cb = static_cast<Int>(b);
        for(Int r : m[a])                 // rows in m[a] only move a -> b in ri
            if (not ri_contains_(ri[static_cast<size_t>(r)], cb)) {
                ri_remove_(ri[static_cast<size_t>(r)], ca);
                ri_insert_(ri[static_cast<size_t>(r)], cb);
            }
        for(Int r : m[b])                 // rows in m[b] only move b -> a in ri
            if (not ri_contains_(ri[static_cast<size_t>(r)], ca)) {
                ri_remove_(ri[static_cast<size_t>(r)], cb);
                ri_insert_(ri[static_cast<size_t>(r)], ca);
            }
        std::swap(m[a], m[b]);
    }

    template<class Int>
    void VRUDecomposition<Int>::row_swap_adjacent_indexed_(MatrixData& m, MatrixData& ri, Int i)
    {
        const Int j = i + 1;
        for(Int c : ri[static_cast<size_t>(i)])      // columns with row i only: relabel i -> j
            if (not ri_contains_(ri[static_cast<size_t>(j)], c)) {
                auto& col = m[static_cast<size_t>(c)];
                *std::lower_bound(col.begin(), col.end(), i) = j;
            }
        for(Int c : ri[static_cast<size_t>(j)])      // columns with row j only: relabel j -> i
            if (not ri_contains_(ri[static_cast<size_t>(i)], c)) {
                auto& col = m[static_cast<size_t>(c)];
                *std::lower_bound(col.begin(), col.end(), j) = i;
            }
        // rows i, j exchange their whole incidence sets
        std::swap(ri[static_cast<size_t>(i)], ri[static_cast<size_t>(j)]);
    }

    template<class Int>
    void VRUDecomposition<Int>::swap_adjacent_rows_(MatrixData& m, Int i)
    {
        const Int j = i + 1;
        for(auto& col : m) {
            auto it_i = std::lower_bound(col.begin(), col.end(), i);
            bool has_i = (it_i != col.end() and *it_i == i);
            auto it_j = std::lower_bound(col.begin(), col.end(), j);
            bool has_j = (it_j != col.end() and *it_j == j);
            if (has_i != has_j) {
                if (has_i) *it_i = j;   // i -> i+1; stays sorted (i+1 absent)
                else       *it_j = i;   // i+1 -> i; stays sorted (i absent)
            }
        }
    }

    // ===================================================================
    // Vineyards: adjacent transposition (Cohen-Steiner; Piekenbrock-Perea
    // Algorithm 6). Conjugation PRP swaps columns and rows i, i+1 of R, V, D;
    // the case-specific S column-adds restore the reduced/upper-triangular
    // invariants of R, V (D only conjugates -- the S factors cancel through
    // PP = I in D' = PDP).
    // ===================================================================

    template<class Int>
    void VRUDecomposition<Int>::transpose(size_t i, DecompositionManipStats* stats)
    {
        check_manip_preconditions_("transpose");

        if (i + 1 >= n_rows)
            throw std::runtime_error("transpose: index out of range");

        // The two cells must be transposable: neither a face of the other.
        // sigma_{i+1} (later) cannot be a face of sigma_i; the only obstruction
        // is sigma_i being a face of sigma_{i+1} (row i present in D column i+1).
        if (std::binary_search(d_data[i + 1].begin(), d_data[i + 1].end(), static_cast<Int>(i)))
            throw std::runtime_error("transpose: cells are in a face relation and cannot be transposed");

        ensure_dynamic_();   // localizes the conjugation to the cells' stars

        Timer timer;
        long long n_add_r = 0, n_add_v = 0, n_queries = 0;

        const bool pos_i = r_data[i].empty();      // i positive (creator)
        const bool pos_j = r_data[i + 1].empty();  // i+1 positive

        auto v_has_i_in_iplus1 = [&]() {
            ++n_queries;
            // V[i][i+1] != 0  <=>  column i+1 is in the row-i incidence list of V
            return ri_contains_(ri_v_[i], static_cast<Int>(i + 1));
        };

        auto add_r = [&](size_t src, size_t dst) { col_add_indexed_(r_data, ri_r_, dst, r_data[src]); ++n_add_r; };
        auto add_v = [&](size_t src, size_t dst) { col_add_indexed_(v_data, ri_v_, dst, v_data[src]); ++n_add_v; };

        auto conjugate = [&]() {
            col_swap_indexed_(r_data, ri_r_, i, i + 1);
            col_swap_indexed_(v_data, ri_v_, i, i + 1);
            col_swap_indexed_(d_data, ri_d_, i, i + 1);
            row_swap_adjacent_indexed_(r_data, ri_r_, static_cast<Int>(i));
            row_swap_adjacent_indexed_(v_data, ri_v_, static_cast<Int>(i));
            row_swap_adjacent_indexed_(d_data, ri_d_, static_cast<Int>(i));
        };

        // R columns whose low may change = stars of rows i, i+1 (plus i, i+1).
        // Snapshot their lows now; recompute _pivots only for these afterward.
        std::vector<size_t> cand{i, i + 1};
        for(Int c : ri_r_[i]) cand.push_back(static_cast<size_t>(c));
        for(Int c : ri_r_[i + 1]) cand.push_back(static_cast<size_t>(c));
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        std::vector<Int> cand_old_low(cand.size());
        for(size_t t = 0; t < cand.size(); ++t)
            cand_old_low[t] = r_data[cand[t]].empty() ? Int(-1) : r_data[cand[t]].back();

        if (pos_i and pos_j) {
            // Case (+,+): both positive (births)
            if (v_has_i_in_iplus1())
                add_v(i, i + 1);

            Int k = _pivots[i];        // column with low_R = i (death of birth i)
            Int l = _pivots[i + 1];    // column with low_R = i+1
            n_queries += 2;
            bool special = (k != -1) and (l != -1)
                and std::binary_search(r_data[l].begin(), r_data[l].end(), static_cast<Int>(i)); // R[i,l] != 0
            ++n_queries;
            if (special) {
                if (k < l) { add_r(k, l); add_v(k, l); }
                else       { add_r(l, k); add_v(l, k); }
            }
            conjugate();
        } else if (not pos_i and not pos_j) {
            // Case (-,-): both negative (deaths)
            if (v_has_i_in_iplus1()) {
                Int low_i = r_data[i].back();
                Int low_j = r_data[i + 1].back();
                add_r(i, i + 1);
                add_v(i, i + 1);
                conjugate();
                if (not (low_i < low_j)) {   // pivots collide after swap -> reduce again
                    add_r(i, i + 1);
                    add_v(i, i + 1);
                }
            } else {
                conjugate();
            }
        } else if (not pos_i and pos_j) {
            // Case (-,+): i negative, i+1 positive
            if (v_has_i_in_iplus1()) {
                add_r(i, i + 1);
                add_v(i, i + 1);
                conjugate();
                add_r(i, i + 1);
                add_v(i, i + 1);
            } else {
                conjugate();
            }
        } else {
            // Case (+,-): i positive, i+1 negative
            if (v_has_i_in_iplus1())
                add_v(i, i + 1);
            conjugate();
        }

        // Incremental pivot update: only the candidate columns' lows can have
        // changed. Clear the stale entries they owned, then set their new lows.
        for(size_t t = 0; t < cand.size(); ++t)
            if (cand_old_low[t] >= 0 and _pivots[cand_old_low[t]] == static_cast<Int>(cand[t]))
                _pivots[cand_old_low[t]] = -1;
        for(size_t c : cand)
            if (not r_data[c].empty())
                _pivots[r_data[c].back()] = static_cast<Int>(c);

        // U (= V^{-1} transposed) is now stale.
        if (not u_data_t.empty())
            u_data_t.clear();
        for(auto& kv : is_elz_in_dim_)
            kv.second = false;

        if (stats) {
            stats->n_transpositions += 1;
            stats->n_column_additions_r += n_add_r;
            stats->n_column_additions_v += n_add_v;
            stats->n_queries += n_queries;
            // localized: columns actually visited (stars of rows i, i+1)
            stats->n_columns_scanned += static_cast<long long>(cand.size()
                    + ri_v_[i].size() + ri_v_[i + 1].size()
                    + ri_d_[i].size() + ri_d_[i + 1].size());
            stats->elapsed_transpose += timer.elapsed();
        }
    }

    template<class Int>
    size_t VRUDecomposition<Int>::transpose_to(const std::vector<size_t>& new_to_old, DecompositionManipStats* stats)
    {
        check_manip_preconditions_("transpose_to");
        const size_t n = r_data.size();
        validate_permutation_(new_to_old, n, "transpose_to");

        Timer t_total;
        if (stats) {
            stats->nnz_r_before = matrix_nnz_(r_data);
            stats->nnz_v_before = matrix_nnz_(v_data);
        }
        // key[p] = target position of the cell currently at position p.
        // Initially position p holds old cell p, whose target is old_to_new[p].
        std::vector<size_t> key = invert_perm(new_to_old);

        // Bubble sort key to ascending via adjacent transpositions; the number
        // of swaps equals the number of inversions (Kendall tau), which is
        // optimal for adjacent transpositions. O(n^2) scan is acceptable for a
        // vineyards baseline (the paper notes vineyards is inherently O(m^2)).
        size_t count = 0;
        bool changed = true;
        while (changed) {
            changed = false;
            for(size_t p = 0; p + 1 < n; ++p) {
                if (key[p] > key[p + 1]) {
                    transpose(p, stats);
                    std::swap(key[p], key[p + 1]);
                    ++count;
                    changed = true;
                }
            }
        }

        if (stats) {
            stats->nnz_r_after = matrix_nnz_(r_data);
            stats->nnz_v_after = matrix_nnz_(v_data);
            stats->elapsed_total += t_total.elapsed();
        }
        return count;
    }

    // ===================================================================
    // Moves (Busaryev; Piekenbrock-Perea 2.4) and move schedules (Alg 4/5).
    //
    // A move is realized as a sequence of |i-j| adjacent transpositions. Once
    // transpose() is localized (O(star) per step via the row index), this
    // costs O(|i-j| * star) and subsumes the Busaryev "donor" conjugate-once
    // approach (whole-matrix O(nnz)) for small/medium moves while matching it
    // for large ones -- and it reuses the validated, index-maintaining
    // transpose, so the index stays consistent across a move with no extra
    // bookkeeping.
    // ===================================================================

    template<class Int>
    void VRUDecomposition<Int>::move_right(size_t i, size_t j, DecompositionManipStats* stats)
    {
        check_manip_preconditions_("move_right");
        if (i == j)
            return;
        if (i > j)
            throw std::runtime_error("move_right requires i < j");
        if (j >= n_rows)
            throw std::runtime_error("move_right: index out of range");

        // Realize the move as |j-i| adjacent transpositions. Each transpose is
        // localized (O(star)) and maintains the row index, so the move costs
        // O((j-i) * star) -- which subsumes the donor conjugate-once approach
        // (whole-matrix O(nnz)) for small/medium moves and matches it for large
        // ones. transpose() fills the transposition/column/scan stats.
        for(size_t p = i; p < j; ++p)
            transpose(p, stats);
        if (stats)
            stats->n_moves += 1;
    }

    template<class Int>
    void VRUDecomposition<Int>::move_left(size_t i, size_t j, DecompositionManipStats* stats)
    {
        check_manip_preconditions_("move_left");
        if (i == j)
            return;
        if (i < j)
            throw std::runtime_error("move_left requires i > j");
        if (i >= n_rows)
            throw std::runtime_error("move_left: index out of range");

        for(size_t p = i; p > j; --p)
            transpose(p - 1, stats);
        if (stats)
            stats->n_moves += 1;
    }

    template<class Int>
    void VRUDecomposition<Int>::move(size_t i, size_t j, DecompositionManipStats* stats)
    {
        if (i < j)
            move_right(i, j, stats);
        else if (i > j)
            move_left(i, j, stats);
    }

    template<class Int>
    size_t VRUDecomposition<Int>::apply_move_schedule(const std::vector<size_t>& new_to_old, DecompositionManipStats* stats)
    {
        check_manip_preconditions_("apply_move_schedule");
        const size_t n = r_data.size();
        validate_permutation_(new_to_old, n, "apply_move_schedule");

        Timer t_total;
        Timer t_sched;
        if (stats) {
            stats->nnz_r_before = matrix_nnz_(r_data);
            stats->nnz_v_before = matrix_nnz_(v_data);
        }

        // Minimal move schedule (Piekenbrock-Perea Alg 4/5). Per dimension,
        // the cells of a longest increasing subsequence of the target-position
        // sequence already sit in correct relative order and never move; every
        // other cell is moved exactly once, so #moves = n - sum|LIS| = the
        // optimal m - |LCS|. The remaining cells are inserted in ascending
        // target order, each placed immediately after the cell that should
        // precede it (target t-1), which keeps the placed cells' relative order
        // correct -- their absolute positions settle as the rotations cascade.
        auto old_to_new = invert_perm(new_to_old);
        // Schedule on the changed support: per dimension, trim the fixed
        // (old_to_new[p] == p) prefix/suffix and run the LIS only over the
        // window that actually moved. Cells outside the window are fixed and
        // cost nothing, so the O(n log n) LIS becomes O(window log window) --
        // the rest of the call is then dominated by the moves themselves.
        std::vector<size_t> order;             // movable cells, collected from the windows
        // Window per MATRIX-space dimension block: _dim_first/_dim_last are the
        // column-index blocks (== dim_first/dim_last for homology; reversed for
        // cohomology). Using these keeps the per-dimension LIS correct in both.
        for(size_t d = 0; d < _dim_first.size(); ++d) {
            size_t blo = static_cast<size_t>(_dim_first[d]);
            size_t bhi = static_cast<size_t>(_dim_last[d]);
            if (bhi < blo)
                continue;
            size_t wlo = blo;
            while (wlo <= bhi and old_to_new[wlo] == wlo)
                ++wlo;
            if (wlo > bhi)
                continue;                       // whole block already in place
            size_t whi = bhi;
            while (old_to_new[whi] == whi)
                --whi;                          // stops at >= wlo (wlo is movable)
            std::vector<size_t> seq;            // target position of each cell in the window
            seq.reserve(whi - wlo + 1);
            for(size_t p = wlo; p <= whi; ++p)
                seq.push_back(old_to_new[p]);
            auto mask = lis_mask(seq);
            for(size_t t = 0; t < seq.size(); ++t)
                if (not mask[t])
                    order.push_back(wlo + t);   // non-LIS window cells move
        }
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return old_to_new[a] < old_to_new[b]; });
        if (stats)
            stats->elapsed_schedule_build += t_sched.elapsed();

        std::vector<size_t> pos(n), cur(n);
        std::iota(pos.begin(), pos.end(), size_t(0)); // pos[p] = old cell at position p
        std::iota(cur.begin(), cur.end(), size_t(0)); // cur[o] = position of old cell o

        size_t n_moves_done = 0;
        for(size_t o : order) {
            size_t t_o = old_to_new[o];
            size_t o_pos = cur[o];
            // destination = right after the predecessor (target t_o - 1) in the
            // current arrangement; t_o == 0 means the very front.
            size_t dest;
            if (t_o == 0) {
                dest = 0;
            } else {
                size_t pred_pos = cur[new_to_old[t_o - 1]];
                dest = (o_pos > pred_pos) ? pred_pos + 1 : pred_pos;
            }
            if (o_pos == dest)
                continue;
            move(o_pos, dest, stats);
            ++n_moves_done;
            if (o_pos < dest) {       // move_right: (o_pos, dest] shift left by one
                size_t moved = pos[o_pos];
                for(size_t p = o_pos; p < dest; ++p) { pos[p] = pos[p + 1]; cur[pos[p]] = p; }
                pos[dest] = moved; cur[moved] = dest;
            } else {                  // move_left: [dest, o_pos) shift right by one
                size_t moved = pos[o_pos];
                for(size_t p = o_pos; p > dest; --p) { pos[p] = pos[p - 1]; cur[pos[p]] = p; }
                pos[dest] = moved; cur[moved] = dest;
            }
        }

        if (stats) {
            stats->nnz_r_after = matrix_nnz_(r_data);
            stats->nnz_v_after = matrix_nnz_(v_data);
            stats->elapsed_total += t_total.elapsed();
        }
        return n_moves_done;
    }

    // ===================================================================
    // Luo-Nelson warm-start updates (Alg 2 / Alg 3).
    // Notation map: their B = our d_data, their U = our v_data, their R = R.
    // ===================================================================

    template<class Int>
    size_t VRUDecomposition<Int>::matrix_nnz_(const MatrixData& m)
    {
        size_t s = 0;
        for(const auto& c : m)
            s += c.size();
        return s;
    }

    template<class Int>
    bool VRUDecomposition<Int>::is_reduced_consistent() const
    {
        using ST = SimpleSparseMatrixTraits<Int, 2>;
        if (not is_matrix_reduced(r_data))
            return false;
        if (v_data.size() != r_data.size() or d_data.size() != r_data.size())
            return false;
        // check D V == R, column by column over Z_2
        for(size_t c = 0; c < r_data.size(); ++c) {
            IntSparseColumn acc;
            for(Int k : v_data[c])
                ST::add_to_column(acc, d_data[static_cast<size_t>(k)]);
            if (acc != r_data[c])
                return false;
        }
        return true;
    }

    template<class Int>
    void VRUDecomposition<Int>::relabel_rows_(MatrixData& m, const std::vector<size_t>& map)
    {
        for(auto& col : m) {
            for(auto& r : col)
                r = static_cast<Int>(map[static_cast<size_t>(r)]);
            std::sort(col.begin(), col.end());
        }
    }

    template<class Int>
    void VRUDecomposition<Int>::update_with_permutation(const std::vector<size_t>& new_to_old, DecompositionManipStats* stats)
    {
        check_manip_preconditions_("update_with_permutation");
        const size_t n = r_data.size();
        validate_permutation_(new_to_old, n, "update_with_permutation");
        invalidate_dynamic_();   // not yet localized: drops the row index

        Timer t_total;
        if (stats) {
            stats->nnz_r_before = matrix_nnz_(r_data);
            stats->nnz_v_before = matrix_nnz_(v_data);
        }

        auto old_to_new = invert_perm(new_to_old);

        // Luo-Nelson Alg 2. Their B=D, U=V, R=R; for a filtration reorder both
        // P_r and P_c^T act as "relabel rows by old_to_new". The column order
        // is NOT fixed up front -- it emerges from re-triangularizing V.
        Timer t_permute;
        // (lines 3-4) relabel rows of R, V and D; columns stay in old order.
        // D V = R is intentionally broken here and restored by the end.
        relabel_rows_(r_data, old_to_new);
        relabel_rows_(v_data, old_to_new);
        relabel_rows_(d_data, old_to_new);
        if (stats)
            stats->elapsed_permute += t_permute.elapsed();

        Timer t_re;
        long long ar = 0, av = 0;

        // (line 5) reduce V to unique pivots, recording column ops into R.
        std::vector<Int> vpiv(n, Int(-1));
        for(size_t c = 0; c < n; ++c)
            reduce_column_with_pivots_(v_data, r_data, c, vpiv, av, ar);

        // (lines 6-9) re-triangularize V: place the column whose V-pivot is j
        // at position j. V is full rank, so vpiv is a permutation of [0,n) and
        // this column reorder lands the decomposition in the new filtration
        // order. d_data is reordered in lockstep so D V == R is restored.
        {
            MatrixData nV(n), nR(n), nD(n);
            for(size_t j = 0; j < n; ++j) {
                Int src = vpiv[j];
                if (src < 0)
                    throw std::runtime_error("update_with_permutation: V not full rank (unexpected)");
                size_t s = static_cast<size_t>(src);
                nV[j] = std::move(v_data[s]);
                nR[j] = std::move(r_data[s]);
                nD[j] = std::move(d_data[s]);
            }
            v_data = std::move(nV);
            r_data = std::move(nR);
            d_data = std::move(nD);
        }

        // (line 11) reduce R to reduced form, recording column ops into V.
        _pivots.assign(n, Int(-1));
        for(size_t c = 0; c < n; ++c)
            reduce_column_with_pivots_(r_data, v_data, c, _pivots, ar, av);
        if (stats)
            stats->elapsed_rereduce += t_re.elapsed();

        is_reduced = true;
        if (not u_data_t.empty())
            u_data_t.clear();
        for(auto& kv : is_elz_in_dim_)
            kv.second = false;

        if (stats) {
            stats->n_column_additions_r += ar;
            stats->n_column_additions_v += av;
            // 3 row-relabels (R, V, D) + 2 reduction passes + 3 column
            // materializations (nR, nV, nD) -- all whole-matrix
            stats->n_columns_scanned += 8 * static_cast<long long>(n);
            stats->nnz_r_after = matrix_nnz_(r_data);
            stats->nnz_v_after = matrix_nnz_(v_data);
            stats->elapsed_total += t_total.elapsed();
        }
    }

    template<class Int>
    void VRUDecomposition<Int>::update_with_edits(const std::vector<long long>& new_to_old,
                                                  const MatrixData& new_boundary,
                                                  const std::vector<Int>& new_dim_first,
                                                  const std::vector<Int>& new_dim_last,
                                                  DecompositionManipStats* stats)
    {
        check_manip_preconditions_("update_with_edits");
        if (dualize_)
            throw std::runtime_error("update_with_edits: cohomology (dualize) not supported yet "
                                     "(insert/delete under the reversed layout is not handled)");
        invalidate_dynamic_();   // resizes; not yet localized
        const size_t n_old = r_data.size();
        const size_t n_new = new_to_old.size();
        if (new_boundary.size() != n_new)
            throw std::runtime_error("update_with_edits: new_boundary size mismatch");

        Timer t_total;
        if (stats) {
            stats->nnz_r_before = matrix_nnz_(r_data);
            stats->nnz_v_before = matrix_nnz_(v_data);
        }

        // survivors in new order; ranks and maps
        std::vector<size_t> survivor_old_by_rank;  // rank -> old index
        std::vector<size_t> srank_to_newpos;       // rank -> new full position
        survivor_old_by_rank.reserve(n_new);
        srank_to_newpos.reserve(n_new);
        for(size_t k = 0; k < n_new; ++k) {
            if (new_to_old[k] >= 0) {
                if (static_cast<size_t>(new_to_old[k]) >= n_old)
                    throw std::runtime_error("update_with_edits: old index out of range");
                survivor_old_by_rank.push_back(static_cast<size_t>(new_to_old[k]));
                srank_to_newpos.push_back(k);
            }
        }
        const size_t n_surv = survivor_old_by_rank.size();
        if (n_surv > n_old)
            throw std::runtime_error("update_with_edits: more survivors than old cells");

        std::vector<char> is_surv(n_old, 0);
        std::vector<size_t> old_to_srank(n_old, 0);
        for(size_t s = 0; s < n_surv; ++s) {
            if (is_surv[survivor_old_by_rank[s]])
                throw std::runtime_error("update_with_edits: old index referenced twice");
            is_surv[survivor_old_by_rank[s]] = 1;
            old_to_srank[survivor_old_by_rank[s]] = s;
        }

        // Step 1: reorder the old decomposition so survivors come first (in new
        // relative order) and deleted cells last (old order); warm via Alg 2.
        Timer t_resize;
        std::vector<size_t> reorder(n_old);
        for(size_t s = 0; s < n_surv; ++s)
            reorder[s] = survivor_old_by_rank[s];
        size_t dp = n_surv;
        for(size_t o = 0; o < n_old; ++o)
            if (not is_surv[o])
                reorder[dp++] = o;

        DecompositionManipStats inner;
        update_with_permutation(reorder, &inner);

        // Step 2: drop the deleted tail. After Alg 2 the survivor columns only
        // reference rows < n_surv, so truncating rows is clean.
        r_data.resize(n_surv);
        v_data.resize(n_surv);
        d_data.resize(n_surv);

        // Step 3: rebuild to the full new order. Survivor reduced columns are
        // relabeled by the order-preserving map rank->new position (keeps V
        // upper-triangular); inserted cells get the new boundary column in R/D
        // and an identity column in V.
        MatrixData nR(n_new), nV(n_new), nD(n_new);
        for(size_t k = 0; k < n_new; ++k) {
            if (new_to_old[k] >= 0) {
                size_t s = old_to_srank[static_cast<size_t>(new_to_old[k])];
                IntSparseColumn rc, vc;
                rc.reserve(r_data[s].size());
                for(Int r : r_data[s]) rc.push_back(static_cast<Int>(srank_to_newpos[static_cast<size_t>(r)]));
                vc.reserve(v_data[s].size());
                for(Int r : v_data[s]) vc.push_back(static_cast<Int>(srank_to_newpos[static_cast<size_t>(r)]));
                // relabel is order-preserving, so already sorted
                nR[k] = std::move(rc);
                nV[k] = std::move(vc);
            } else {
                nR[k] = new_boundary[k];
                nV[k] = IntSparseColumn{static_cast<Int>(k)};
            }
            nD[k] = new_boundary[k];
        }
        r_data = std::move(nR);
        v_data = std::move(nV);
        d_data = std::move(nD);
        n_rows = n_new;
        // dimension layout of the new filtration (homology: _dim == dim)
        if (new_dim_first.size() != new_dim_last.size())
            throw std::runtime_error("update_with_edits: dim arrays size mismatch");
        dim_first = new_dim_first;
        dim_last = new_dim_last;
        _dim_first = new_dim_first;
        _dim_last = new_dim_last;
        is_elz_in_dim_.clear();
        for(dim_type d = 0; d < dim_first.size(); ++d)
            is_elz_in_dim_[d] = false;
        if (stats)
            stats->elapsed_resize += t_resize.elapsed();

        // Step 4: re-reduce R over the warm basis (V stays upper-triangular).
        Timer t_re;
        long long ar = 0, av = 0;
        _pivots.assign(n_new, Int(-1));
        for(size_t c = 0; c < n_new; ++c)
            reduce_column_with_pivots_(r_data, v_data, c, _pivots, ar, av);
        if (stats)
            stats->elapsed_rereduce += t_re.elapsed();

        is_reduced = true;
        if (not u_data_t.empty())
            u_data_t.clear();

        if (stats) {
            stats->n_column_additions_r += inner.n_column_additions_r + ar;
            stats->n_column_additions_v += inner.n_column_additions_v + av;
            // inner Alg-2 reorder + rebuild materialization (nR, nV, nD) + reduce-R
            stats->n_columns_scanned += inner.n_columns_scanned + 4 * static_cast<long long>(n_new);
            stats->nnz_r_after = matrix_nnz_(r_data);
            stats->nnz_v_after = matrix_nnz_(v_data);
            stats->elapsed_total += t_total.elapsed();
        }
    }

    template<class Int>
    template<class WorkCol>
    void VRUDecomposition<Int>::reduce_parallel_r_only_impl(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;
        using namespace std::placeholders;

        size_t n_cols = size();

        if (n_cols == 0)
            return;

        using MatrixTraits = GenericSparseMatrixTraits<Int, WorkCol>;
        using Column = typename MatrixTraits::Column;
        using AMatrix = std::vector<typename MatrixTraits::APColumn>;
        using MemoryReclaimC = MemoryReclaim<Column>;

        std::atomic<typename MemoryReclaimC::EpochCounter> counter;
        counter = 0;

        std::atomic<int> next_free_chunk = 0;
        std::vector<std::atomic<int>> next_free_chunks(_dim_first.size());
        for(auto& nfc : next_free_chunks) {
            nfc.store(0, std::memory_order_relaxed);
        }

        const int n_threads = std::min(params.n_threads, std::max(1, static_cast<int>(n_cols / params.chunk_size)));

        Timer timer_prepare;

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
        params.timings.prepare = timer_prepare.elapsed();

        spd::debug("Pivots initialized");


        std::vector<std::thread> ts;
        std::vector<std::unique_ptr<MemoryReclaimC>> mms;
        std::vector<ThreadStats> stats;

        mms.reserve(n_threads);
        stats.reserve(n_threads);

        Timer timer_reduction;

        for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

            mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
            stats.emplace_back(thread_idx);

            ts.emplace_back(parallel_reduction<MatrixTraits, Int, MemoryReclaimC>,
                    std::ref(ar_matrix), std::ref(pivots), std::ref(next_free_chunk),
                    params, thread_idx, mms[thread_idx].get(), std::ref(stats[thread_idx]),
                    std::ref(next_free_chunks), std::ref(_dim_first), std::ref(_dim_last));

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

        // Workers in parallel_reduction set-and-return on interrupt; the
        // orchestrator throws on the joining thread.
        if (oineus::interrupted())
            throw oineus::interrupted_exception{};

        params.timings.reduce = timer_reduction.elapsed();

        if (params.print_time) {
            long total_cleared = 0;
            for(const auto& s: stats) {
                total_cleared += s.n_cleared;
                spd::info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots);
            }
            spd::info("n_threads = {}, chunk = {}, total_cleared = {}, elapsed = {} sec", n_threads, params.chunk_size, total_cleared, params.timings.reduce);
        }

#ifdef OINEUS_GATHER_ADD_STATS
        write_add_stats_file(stats);
#endif
        {
            Timer timer_copy_back;
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
            params.timings.copy_back = timer_copy_back.elapsed();
        }

        {
            Timer timer_copy_pivots;
            tf::Taskflow taskflow_copy_pivots;
            _pivots.clear();
            _pivots.resize(r_data.size());
            taskflow_copy_pivots.for_each_index((size_t)0, n_cols, (size_t)1,
                    [this, &pivots](size_t col_idx) {
                        _pivots[col_idx] = pivots[col_idx].load(std::memory_order_relaxed);
                    });
            executor.run(taskflow_copy_pivots).get();
            params.timings.copy_pivots = timer_copy_pivots.elapsed();
        }

        // Free the working columns retired during reduction (deferred by the
        // hazard-pointer reclaimers). This is real teardown of the working
        // matrix, so account it under copy_back rather than leaving it untimed
        // at scope exit. Safe now that all worker threads have joined.
        {
            Timer timer_teardown;
            mms.clear();
            params.timings.copy_back += timer_teardown.elapsed();
        }

        is_reduced = true;
    }

    template<class Int>
    template<class WorkCol>
    void VRUDecomposition<Int>::reduce_parallel_rv_impl(Params& params)
    {
        CALI_CXX_MARK_FUNCTION;
        using namespace std::placeholders;

        size_t n_cols = size();

        if (n_cols == 0)
            return;

        int n_threads = std::min(params.n_threads, std::max(1, static_cast<int>(n_cols / params.chunk_size)));

        Timer timer_prepare;

        v_data = std::vector<IntSparseColumn>(n_cols);

        using MatrixTraits = GenericRVMatrixTraits<Int, WorkCol>;

        using RVColumn = typename MatrixTraits::Column;
        using RVMatrix = std::vector<typename MatrixTraits::APColumn>;
        using MemoryReclaimC = MemoryReclaim<RVColumn>;

        RVMatrix r_v_matrix(n_cols);

        std::atomic<typename MemoryReclaimC::EpochCounter> counter;
        counter = 0;

        std::atomic<int> next_free_chunk = 0;
        std::vector<std::atomic<int>> next_free_chunks(_dim_first.size());
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
        params.timings.prepare = timer_prepare.elapsed();

        std::vector<std::thread> ts;
        std::vector<std::unique_ptr<MemoryReclaimC>> mms;
        std::vector<ThreadStats> stats;

        mms.reserve(n_threads);
        stats.reserve(n_threads);

        next_free_chunk = 0;

        Timer timer_reduction;

        for(int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {

            mms.emplace_back(new MemoryReclaimC(n_threads, counter, thread_idx));
            stats.emplace_back(thread_idx);

            ts.emplace_back(parallel_reduction<MatrixTraits, Int, MemoryReclaimC>,
                    std::ref(r_v_matrix), std::ref(pivots), std::ref(next_free_chunk),
                    params, thread_idx, mms[thread_idx].get(), std::ref(stats[thread_idx]),
                    std::ref(next_free_chunks), std::ref(_dim_first),
                    std::ref(_dim_last));

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

        if (oineus::interrupted())
            throw oineus::interrupted_exception{};

        params.timings.reduce = timer_reduction.elapsed();

        if (params.print_time) {
            long total_cleared = 0;
            for(const auto& s: stats) {
                total_cleared += s.n_cleared;
                spd::info("Thread {}: cleared {}, right jumps {}", s.thread_id, s.n_cleared, s.n_right_pivots);
            }
            spd::info("n_threads = {}, chunk = {}, total_cleared = {}, elapsed = {} sec", n_threads, params.chunk_size, total_cleared, params.timings.reduce);
        }

#ifdef OINEUS_GATHER_ADD_STATS
        write_add_stats_file(stats);
#endif
        const size_t n_workers = std::max<size_t>(1, std::min(static_cast<size_t>(n_threads), n_cols));

        // Deterministic static partitioning to avoid a hot global fetch_add in tight loops.

        auto run_parallel_cols = [this, n_workers](dim_type dim, const auto& fn) {
            if (dualize()) {
                dim = _dim_first.size() - dim - 1;
            }
            const size_t start_idx = static_cast<size_t>(_dim_first[dim]);
            const size_t end_idx   = static_cast<size_t>(_dim_last[dim]) + 1;
            const size_t n_cols_in_dim = end_idx - start_idx;
            if (n_cols_in_dim == 0)
                return;

            const size_t n_workers_eff = std::min(n_workers, n_cols_in_dim);

            std::vector<std::thread> workers;
            workers.reserve(n_workers_eff);

            for (size_t tid = 0; tid < n_workers_eff; ++tid) {
                const size_t begin = start_idx + (tid * n_cols_in_dim) / n_workers_eff;
                const size_t end   = start_idx + ((tid + 1) * n_cols_in_dim) / n_workers_eff;
                workers.emplace_back([begin, end, &fn]() {
                    for (size_t col_idx = begin; col_idx < end; ++col_idx) {
                        fn(col_idx);
                    }
                });
            }

            for (auto& t : workers) {
                t.join();
            }
        };

        // auto run_parallel_cols = [n_cols, n_workers](const auto& fn) {
        //     std::vector<std::thread> workers;
        //     workers.reserve(n_workers);
        //
        //     for (size_t tid = 0; tid < n_workers; ++tid) {
        //         const size_t begin = (tid * n_cols) / n_workers;
        //         const size_t end = ((tid + 1) * n_cols) / n_workers;
        //         workers.emplace_back([begin, end, &fn]() {
        //             for (size_t col_idx = begin; col_idx < end; ++col_idx) {
        //                 fn(col_idx);
        //             }
        //         });
        //     }
        //
        //     for (auto& t : workers) {
        //         t.join();
        //     }
        // };


        if (params.dims_to_restore_elz.size() > 0) {
            Timer timer_restore;

            // Step 1: Bauer-trick fill for ALL cleared columns across
            // every dim. The downstream copy-back walks every dim and
            // expects every column pointer to be non-null, so we must
            // fill cleared columns everywhere, not just in the dims
            // we will restore_elz on.
            {
                std::atomic<Int> missing_bauer_col{-1};
                for (dim_type d = 0;
                     d < static_cast<dim_type>(_dim_first.size()); ++d) {
                    run_parallel_cols(dualize() ? _dim_first.size() - d - 1 : d,
                                      [&](size_t col_idx) {
                        auto p = r_v_matrix[col_idx].load(std::memory_order_relaxed);
                        if (p != nullptr) return;
                        const Int pivot_col = pivots[col_idx].load(std::memory_order_relaxed);
                        if (pivot_col < 0) {
                            Int expected = -1;
                            missing_bauer_col.compare_exchange_strong(expected, static_cast<Int>(col_idx),
                                    std::memory_order_relaxed, std::memory_order_relaxed);
                            return;
                        }
                        auto pivot_ptr = r_v_matrix[pivot_col].load(std::memory_order_relaxed);
                        if (pivot_ptr == nullptr) {
                            Int expected = -1;
                            missing_bauer_col.compare_exchange_strong(expected, static_cast<Int>(col_idx),
                                    std::memory_order_relaxed, std::memory_order_relaxed);
                            return;
                        }
                        IntSparseColumn filled_v_col(pivot_ptr->r_column);
                        auto* new_col = new RVColumn(IntSparseColumn(), std::move(filled_v_col));
                        r_v_matrix[col_idx].store(new_col, std::memory_order_relaxed);
                    });
                }
                if (missing_bauer_col.load(std::memory_order_relaxed) >= 0) {
                    throw std::runtime_error("Bauer trick failed while filling V in reduce_parallel_rv");
                }
            }

            // Step 2: snapshot pointers (for double-free-safe reclaim
            // after restore_elz may swap pointers below).
            std::vector<RVColumn*> r_v_matrix_copy(n_cols, nullptr);
            for(size_t col_idx = 0; col_idx < n_cols; ++col_idx) {
                r_v_matrix_copy[col_idx] = r_v_matrix[col_idx].load(std::memory_order_relaxed);
            }

            // Step 3: ELZ restore over the requested dims (others stay
            // unrestored but still have valid Bauer-filled V columns).
            for(dim_type dim: params.dims_to_restore_elz) {
                run_parallel_cols(dim, [&](size_t col_idx) {
                    restore_elz_column_parallel<Int>(r_v_matrix, col_idx);
                });
                // is_elz_in_dim_ uses the internal _dim key (matrix layout).
                const dim_type _dim = static_cast<dim_type>(_dim_from_dim(dim));
                set_is_elz_flag(_dim, true);
            }
            params.timings.restore_elz = timer_restore.elapsed();

            // Step 4: copy back from r_v_matrix to r_data / v_data.
            Timer timer_copy_back;
            for(int dim_idx = _dim_first.size() - 1; dim_idx >= 0; --dim_idx) {
                for(Int col_idx = _dim_first[dim_idx]; col_idx <= _dim_last[dim_idx]; ++col_idx) {
                    auto p = r_v_matrix[col_idx].load(std::memory_order_relaxed);
                    if (p == nullptr) {
                        throw std::runtime_error("NULL column after restore_elz in reduce_parallel_rv");
                    }
                    r_data[col_idx] = std::move(p->r_column);
                    v_data[col_idx] = std::move(p->v_column);
                    if (r_data[col_idx].size() > 0) {
                        if (pivots[r_data[col_idx].back()] != col_idx) {
                            IC(col_idx);
                            IC(pivots[r_data[col_idx].back()]);
                            IC(r_data[col_idx]);
                            throw std::runtime_error("pivots[low(r_data[col_idx])] != col_idx");
                        }
                    }
                    if (v_data[col_idx].empty() or v_data[col_idx].back() != col_idx) {
                        IC(col_idx);
                        IC(v_data[col_idx]);
                        throw std::runtime_error("V column is not 1-diag");
                    }
                }
            }
            for(size_t col_idx = 0; col_idx < n_cols; ++col_idx) {
                auto p_current = r_v_matrix[col_idx].load(std::memory_order_relaxed);
                auto p_original = r_v_matrix_copy[col_idx];
                if (p_current == p_original) {
                    delete p_current;
                } else {
                    delete p_current;
                    delete p_original;
                }
            }
            params.timings.copy_back = timer_copy_back.elapsed();

        } else {
            Timer timer_copy_back;
            for(int dim_idx = _dim_first.size() - 1; dim_idx >= 0; --dim_idx) {
                for(Int col_idx = _dim_first[dim_idx]; col_idx <= _dim_last[dim_idx]; ++col_idx) {
                    auto p = r_v_matrix[col_idx].load(std::memory_order_relaxed);
                    if (p) {
                        r_data[col_idx] = std::move(p->r_column);
                        v_data[col_idx] = std::move(p->v_column);
                        if (r_data[col_idx].size() > 0) {
                            if (pivots[r_data[col_idx].back()] != col_idx) {
                                IC(col_idx);
                                IC(pivots[r_data[col_idx].back()]);
                                IC(r_data[col_idx]);
                                throw std::runtime_error("pivots[low(r_data[col_idx])] != col_idx");
                            }
                        }
                        delete p;
                    } else {
                        // column was cleared
                        r_data[col_idx].clear();
                        // Bauer's trick with filling V
                        v_data[col_idx] = r_data.at(pivots.at(col_idx));
                    }
                    if (v_data[col_idx].empty() or v_data[col_idx].back() != col_idx) {
                        IC(col_idx);
                        IC(pivots.at(col_idx));
                        IC(v_data[col_idx]);
                        throw std::runtime_error("V column is not 1-diag");
                    }
                } // loop over columns
            } // loop over dimensions
            params.timings.copy_back = timer_copy_back.elapsed();
        }

        {
            Timer timer_copy_pivots;
            tf::Taskflow taskflow_copy_pivots;
            _pivots.clear();
            _pivots.resize(r_data.size());
            taskflow_copy_pivots.for_each_index((size_t)0, n_cols, (size_t)1,
                    [this, &pivots](size_t col_idx) {
                        _pivots[col_idx] = pivots[col_idx].load(std::memory_order_relaxed);
                    });
            executor.run(taskflow_copy_pivots).get();
            params.timings.copy_pivots = timer_copy_pivots.elapsed();
        }

        // Free the working columns retired during reduction (deferred by the
        // hazard-pointer reclaimers). This is real teardown of the working
        // matrix, so account it under copy_back rather than leaving it untimed
        // at scope exit. Safe now that all worker threads have joined.
        {
            Timer timer_teardown;
            mms.clear();
            params.timings.copy_back += timer_teardown.elapsed();
        }

        is_reduced = true;
    }

    template<class Int>
    template<class Cell, class Real>
    Diagrams<Real> VRUDecomposition<Int>::diagram_general(const Filtration<Cell, Real>& fil, bool include_all, bool include_inf_points, bool only_zero_persistence) const
    {
        if (not is_reduced)
            throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

        Diagrams<Real> result = fil.size() == 0 ? Diagrams<Real>() : Diagrams<Real>(fil.max_dim());

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

                if (i % 100 == 0 && oineus::interrupted())
                    throw oineus::interrupted_exception{};
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

            if (col_idx % 100 == 0 && oineus::interrupted())
                throw oineus::interrupted_exception{};
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

        Diagrams<Real> result = fil.size() == 0 ? Diagrams<Real>() : Diagrams<Real>(fil.max_dim());

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

                if (i % 100 == 0 && oineus::interrupted())
                    throw oineus::interrupted_exception{};
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

            if (col_idx % 100 == 0 && oineus::interrupted())
                throw oineus::interrupted_exception{};
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

    template<typename Int_>
    typename VRUDecomposition<Int_>::IntSparseColumn
    VRUDecomposition<Int_>::compute_u_column(size_t col_idx) const
    {
        using MatrixTraits = SimpleSparseMatrixTraits<Int_, 2>;

        if (not is_reduced)
            throw std::runtime_error("Cannot compute U column from non-reduced decomposisition");

        if (not has_matrix_v())
            throw std::runtime_error("Cannot compute U column from non-reduced decomposisition");

        IntSparseColumn result;

        auto residual = MatrixTraits::load_to_cache(d_data.at(col_idx));

        while (not MatrixTraits::is_zero(residual)) {
            auto low_idx = MatrixTraits::low(residual);
            auto piv_col_idx = _pivots.at(low_idx);
            result.push_back(piv_col_idx);
            MatrixTraits::add_to_cached(r_data[piv_col_idx], residual);
        }

        std::sort(result.begin(), result.end());

        if (result.empty() or result.back() < static_cast<Int>(col_idx)) {
            if (!r_data.at(col_idx).empty())
                throw std::runtime_error("diagonal problem");
            result.push_back(col_idx);
        }

        return result;
    }

    template<typename Int_>
    typename VRUDecomposition<Int_>::IntSparseColumn
    VRUDecomposition<Int_>::compute_u_column_1(size_t col_idx) const
    {
        using MatrixTraits = SimpleSparseMatrixTraits<Int_, 2>;

        if (not is_reduced)
            throw std::runtime_error("Cannot compute U column from non-reduced decomposisition");

        if (not has_matrix_v())
            throw std::runtime_error("Cannot compute U column from non-reduced decomposisition");

        IntSparseColumn result;

        auto residual = MatrixTraits::cached_identity_column(col_idx);

        while (not MatrixTraits::is_zero(residual)) {
            // V is upper triangular: low and pivot of a column are equal
            auto piv_col_idx = MatrixTraits::low(residual);
            result.push_back(piv_col_idx);
            MatrixTraits::add_to_cached(v_data[piv_col_idx], residual);
        }

        if (result.empty()) {
            result.push_back(col_idx);
        } else {
            std::sort(result.begin(), result.end());
        }

        return result;
    }

    template<typename Int_>
    void VRUDecomposition<Int_>::compute_u_from_v_1(dim_type dim, size_t n_threads, bool verbose)
    {
        Timer timer;
        using MatrixTraits = SimpleSparseMatrixTraits<Int_, 2>;

        // compute columns of U in parallel
        MatrixData u_data = MatrixData(v_data.size());

        const size_t _dim = _dim_from_dim(dim);

        const size_t col_start = range_start_(_dim);
        const size_t col_end = range_end_(_dim);

        // for(size_t col_idx = col_start; col_idx < col_end; ++col_idx) {
        //     u_data[col_idx] = compute_u_column(col_idx);
        // }

        if (verbose) IC(col_start, col_end);

        std::atomic<size_t> next_free_column(col_start);
        std::vector<std::thread> workers;
        workers.reserve(n_threads);

        for(size_t tid = 0; tid < n_threads; ++tid) {
            workers.emplace_back([this, &u_data, &next_free_column, col_end]() {
                while(true) {
                    const size_t col_idx = next_free_column.fetch_add(1, std::memory_order_relaxed);
                    if (col_idx >= col_end)
                        break;
                    u_data[col_idx] = compute_u_column_1(col_idx);
                }
            });
        }

        for(auto& worker: workers)
            worker.join();

        auto col_inv_elapsed = timer.elapsed_reset();

        u_data_t = MatrixTraits::col_to_row_format_parallel(u_data, n_threads, col_start, col_end, v_data.size());

        auto col_to_row_elapsed = timer.elapsed_reset();

        if (verbose) IC(col_inv_elapsed, col_to_row_elapsed);
    }

    template<typename Int_>
    void VRUDecomposition<Int_>::compute_u_from_v(dim_type dim, size_t n_threads, bool verbose)
    {
        Timer timer;
        using MatrixTraits = SimpleSparseMatrixTraits<Int_, 2>;

        // compute columns of U in parallel
        MatrixData u_data = MatrixData(v_data.size());

        const auto _dim = _dim_from_dim(dim);
        const size_t col_start = range_start_(_dim);
        const size_t col_end = range_end_(_dim);

        std::atomic<size_t> next_free_column(col_start);
        std::vector<std::thread> workers;
        workers.reserve(n_threads);

        for(size_t tid = 0; tid < n_threads; ++tid) {
            workers.emplace_back([this, &u_data, &next_free_column, col_end]() {
                while(true) {
                    const size_t col_idx = next_free_column.fetch_add(1, std::memory_order_relaxed);
                    if (col_idx >= col_end)
                        break;
                    u_data[col_idx] = compute_u_column(col_idx);
                }
            });
        }

        for(auto& worker: workers)
            worker.join();

        auto col_inv_elapsed = timer.elapsed_reset();

        u_data_t = MatrixTraits::col_to_row_format_parallel(u_data, n_threads, col_start, col_end, v_data.size());

        auto col_to_row_elapsed = timer.elapsed_reset();

        if (verbose) IC(col_inv_elapsed, col_to_row_elapsed);
    }

    template<typename Int_>
    template<typename Real, typename ValueAt, typename CmpOp>
    typename VRUDecomposition<Int_>::IntSparseColumn
    VRUDecomposition<Int_>::compute_u_row_bounded(size_t row_idx,
                                                  const MatrixData& vt_data,
                                                  Real value_bound,
                                                  ValueAt&& value_at,
                                                  CmpOp&& cmp_op) const
    {
        // Row-form analogue of compute_u_column_1_bounded. Solves
        // (row r of U) * V = e_r^T via residual-style forward
        // substitution against V^T (lower unit-triangular). Pivots
        // strictly increase in matrix index because vt_data[p]'s
        // smallest entry is p (V[p][p] = 1) and XOR cancels it; the
        // remaining entries are all > p.
        using MatrixTraits = SimpleSparseMatrixTraits<Int_, 2>;

        if (not is_reduced)
            throw std::runtime_error("Cannot compute U row from non-reduced decomposition");
        if (not has_matrix_v())
            throw std::runtime_error("Cannot compute U row from non-reduced decomposition");

        IntSparseColumn result;
        auto residual = MatrixTraits::cached_identity_column(row_idx);

        while (not MatrixTraits::is_zero(residual)) {
            auto piv_col_idx = MatrixTraits::top(residual);
            if (cmp_op(value_at(piv_col_idx), value_bound)) {
                break;
            }
            // Defensive fence: if vt_data[piv] is empty, residual will
            // not advance and we would loop forever (with result growing
            // without bound -> memory exhaustion / OS kill). This means
            // V[piv][piv] = 1 is not satisfied -- typically because
            // clearing zeroed the diagonal for this column on the
            // forward side and restore_elz did not put it back. Bail out
            // with a clear error rather than hang.
            if (static_cast<size_t>(piv_col_idx) >= vt_data.size()
                or vt_data[piv_col_idx].empty()) {
                std::ostringstream dbg;
                if (static_cast<size_t>(piv_col_idx) < v_data.size()) {
                    dbg << " v_data[" << piv_col_idx << "].size()="
                        << v_data[piv_col_idx].size() << " contents:";
                    for (auto x : v_data[piv_col_idx]) dbg << " " << x;
                }
                throw std::runtime_error(
                    "compute_u_row_bounded: vt_data["
                    + std::to_string(piv_col_idx) + "] is empty; "
                    "V does not have unit diagonal at this column."
                    + dbg.str());
            }
            result.push_back(piv_col_idx);
            MatrixTraits::add_to_cached(vt_data[piv_col_idx], residual);
        }

        if (result.empty()) {
            // Diagonal-element invariant: U[r][r] = 1 always; the
            // walker's assert(not result.empty()) requires u_data_t[r]
            // to contain r. A degenerate bound that ruled out the
            // diagonal still falls back here.
            result.push_back(row_idx);
        }
        // result is sorted ascending by construction (pivots strictly
        // increase), so no explicit sort needed.

        return result;
    }

    template<typename Int_>
    template<typename Real, typename ValueAt, typename CmpOp>
    void VRUDecomposition<Int_>::compute_partial_u_rows(
            const std::vector<size_t>& rows,
            const std::vector<Real>& bounds,
            dim_type dim,
            ValueAt&& value_at,
            CmpOp&& cmp_op,
            size_t n_threads,
            bool verbose)
    {
        if (rows.size() != bounds.size())
            throw std::runtime_error("compute_partial_u_rows: rows and bounds must have the same size");

        if (u_data_t.size() != v_data.size()) {
            u_data_t = MatrixData(v_data.size());
        }

        if (rows.empty())
            return;

        const auto _dim = _dim_from_dim(dim);

        // The row solver assumes V is in ELZ form. Silent drift here
        // yields wrong gradients; consult the cached flag. A full
        // is_elz() walk would be O(matrix) per call, far too
        // expensive for the gradient loop. The flag is set by
        // restore_elz() and by serial reduction without clearing.
        if (not is_elz_in_dim_.at(_dim))
            throw std::runtime_error(
                "compute_partial_u_rows: V is not known to be in ELZ "
                "form. Call restore_elz() or reduce serially without "
                "clearing before this entry point.");

        if (n_threads == 0) n_threads = 1;
        n_threads = std::min(n_threads, rows.size());

        Timer timer;
        using MatrixTraits = SimpleSparseMatrixTraits<Int_, 2>;

        // Stage A: build vt_data restricted to dim d (parallel transpose).
        // V is block-diagonal in dim, so transposing only the dim's
        // matrix-column range gives us exactly the rows of V^T that
        // any row solve in dim d will read.
        auto vt_data = MatrixTraits::col_to_row_format_parallel(
                v_data, static_cast<int>(n_threads),
                range_start_(_dim), range_end_(_dim),
                static_cast<typename MatrixTraits::Int>(v_data.size()));

        auto vt_elapsed = timer.elapsed_reset();

        // Stage B: parallel row solves. Each row writes to its own
        // u_data_t[r] slot; no shared writes.
        std::atomic<size_t> next_free(0);
        std::vector<std::thread> workers;
        workers.reserve(n_threads);

        for (size_t tid = 0; tid < n_threads; ++tid) {
            workers.emplace_back([this, &rows, &bounds, &vt_data,
                                  &next_free, &value_at, &cmp_op]() {
                while (true) {
                    const size_t i = next_free.fetch_add(1, std::memory_order_relaxed);
                    if (i >= rows.size()) break;
                    u_data_t[rows[i]] = compute_u_row_bounded(
                            rows[i], vt_data, bounds[i], value_at, cmp_op);
                }
            });
        }

        for (auto& w : workers) w.join();

        auto solve_elapsed = timer.elapsed_reset();

        if (verbose) IC(vt_elapsed, solve_elapsed);
    }

    template<typename Int_>
    template<typename Real, typename ValueAt>
    void VRUDecomposition<Int_>::compute_full_u_rows(dim_type dim,
                                                     ValueAt&& value_at,
                                                     size_t n_threads,
                                                     bool verbose)
    {
        const auto _dim = _dim_from_dim(dim);
        const size_t cstart = range_start_(_dim);
        const size_t cend = range_end_(_dim);
        if (cend <= cstart) {
            if (u_data_t.size() != v_data.size())
                u_data_t = MatrixData(v_data.size());
            return;
        }
        std::vector<size_t> rows;
        rows.reserve(cend - cstart);
        for (size_t r = cstart; r < cend; ++r) rows.push_back(r);
        std::vector<Real> bounds(rows.size(),
                                 std::numeric_limits<Real>::max());
        auto never_stop = [](Real, Real) { return false; };
        compute_partial_u_rows(rows, bounds, dim, value_at, never_stop,
                               n_threads, verbose);
    }

    template<typename Int>
    std::ostream& operator<<(std::ostream& out, const VRUDecomposition<Int>& m)
    {
        using Traits = SimpleSparseMatrixTraits<Int, 2>;
        using MatrixData = typename VRUDecomposition<Int>::MatrixData;

        out << "Decomposition(size=" << m.r_data.size()
            << ", dualize=" << (m.dualize() ? "true" : "false")
            << ", reduced=" << (m.is_reduced ? "true" : "false")
            << ", has_V=" << (m.has_matrix_v() ? "true" : "false")
            << ", has_U=" << (m.has_matrix_u() ? "true" : "false")
            << ")\n";

        auto print_matrix = [&out](const char* tag, const char* line_tag,
                                   const MatrixData& mat) {
            out << "Matrix " << tag << "[\n";
            for(size_t idx = 0; idx < mat.size(); ++idx) {
                out << line_tag << " " << idx << ": ";
                Traits::print_column(out, mat[idx]);
                out << "\n";
            }
            out << "]\n";
        };

        print_matrix("D", "Column", m.d_data);
        print_matrix("R", "Column", m.r_data);
        print_matrix("V", "Column", m.v_data);
        print_matrix("U", "Row",    m.u_data_t);

        return out;
    }
}
