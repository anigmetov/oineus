#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <ostream>

#include <algorithm>

#include <taskflow/taskflow.hpp>
#include <taskflow/core/flow_builder.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <taskflow/algorithm/for_each.hpp>

#include "timer.h"
#include "simplex.h"
#include "decomposition.h"
#include "params.h"
namespace oineus {

    template<typename Int_, typename Real_, size_t D>
    class Grid;

    // filtration + vector of locations of critical value for each simplex
    // L = int: vertex id in grid for lower-star
    // L = VREdge: edge for Vietoris--Rips
    template<typename Cell, typename Real, typename L>
    struct FiltrationCritValLocs {
        Filtration<Cell, Real> filtration;
        std::vector<L> critical_value_locations;
    };

    template<class UnderCell_, class Real_>
    class Filtration {
    public:
        using Real = Real_;
        using Cell = CellWithValue<UnderCell_, Real>;
        using UidHasher = typename UnderCell_::UidHasher;
        using Int = typename Cell::Int;
        using CellVector = std::vector<Cell>;
        using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;

        using CellUid = typename Cell::Uid;

        Filtration() = default;
        Filtration(const Filtration&) = default;
        Filtration(Filtration&&) = default;
        Filtration& operator=(const Filtration&) = default;
        Filtration& operator=(Filtration&&) = default;

        // use move constructor to move cells
        Filtration(CellVector&& _simplices, bool _negate, int n_threads = 1, bool _sort_only_by_dim=false, bool _set_uids=true)
                :
                negate_(_negate),
                cells_(std::move(_simplices)),
                id_to_sorted_id_(cells_.size()),
                sorted_id_to_id_(cells_.size())
        {
            CALI_CXX_MARK_FUNCTION;
            init(n_threads, _sort_only_by_dim, _set_uids);
        }

        void init(int n_threads, bool sort_only_by_dim, bool _set_uids)
        {
            CALI_CXX_MARK_FUNCTION;

            if (sort_only_by_dim) {
                sort_dim_only();
            } else {
                sort_and_set(n_threads, _set_uids);
            }

            set_dim_info();

            assert(std::all_of(cells_.begin(), cells_.end(),
                    [](const Cell& sigma) { return sigma.is_valid_filtration_simplex(); }));
        }

        void reset_ids_to_sorted_ids()
        {
            CALI_CXX_MARK_FUNCTION;
            for(auto& sigma : cells_) {
                sigma.set_id(sigma.get_sorted_id());
            }
            for(size_t i = 0; i < id_to_sorted_id_.size(); ++i) {
                id_to_sorted_id_[i] = i;
                sorted_id_to_id_[i] = i;
            }
        }

        // copy cells
        Filtration(const CellVector& cells, bool _negate, int n_threads = 1, bool _sort_only_by_dim=false, bool _set_ids=true)
                :
                negate_(_negate),
                cells_(cells),
                id_to_sorted_id_(cells.size()),
                sorted_id_to_id_(cells.size())
        {
            init(n_threads, _sort_only_by_dim, _set_ids);
        }

        size_t size() const { return cells_.size(); }

        auto dim_first(dim_type d) const { return dim_first_.at(d); }
        auto dim_last(dim_type d) const { return dim_last_.at(d); }

        auto dim_first() const { return dim_first_; }
        auto dim_last() const { return dim_last_; }

        Real get_cell_value(size_t i) const { return cells_[i].get_value(); }

        size_t size_in_dimension(dim_type d) const
        {
            if (d > max_dim())
                return 0;
            Int result = dim_last(d) - dim_first(d) + 1;
            if (result < 0)
                throw std::runtime_error("dim_last less than dim_first");
            return static_cast<size_t>(result);
        }

        size_t n_vertices() const { return size_in_dimension(0); }

        dim_type max_dim() const { return dim_last_.size() - 1; }

        BoundaryMatrix boundary_matrix(int n_threads=1) const
        {
            CALI_CXX_MARK_FUNCTION;
            BoundaryMatrix result(size());

            bool missing_ok = is_subfiltration();

            // boundary of vertex is empty, need to do something in positive dimension only
            tf::Executor executor(n_threads);
            tf::Taskflow taskflow;
            taskflow.for_each_index((size_t)0, size(), (size_t)1,
                    [this, missing_ok, &result](size_t col_idx) {
                        // skip 0-dim simplices
                        if (col_idx <= static_cast<size_t>(dim_last(0)))
                            return;
                        const auto& sigma = cells_[col_idx];
                        auto& col = result[col_idx];
                        col.reserve(sigma.dim() + 1);

                        for(const auto& tau_vertices: sigma.boundary()) {
                            if (missing_ok) {
                                auto iter = uid_to_sorted_id.find(tau_vertices);
                                if (iter != uid_to_sorted_id.end()) {
                                    col.push_back(iter->second);
                                }
                            } else {
                                col.push_back(uid_to_sorted_id.at(tau_vertices));
                            }
                        }
                        std::sort(col.begin(), col.end());
                    });
            executor.run(taskflow).get();
            return result;
        }

        BoundaryMatrix boundary_matrix_in_dimension(dim_type d, int n_threads) const
        {
            CALI_CXX_MARK_FUNCTION;
            bool missing_ok = is_subfiltration();

            BoundaryMatrix result(size_in_dimension(d));
            // fill D with empty vectors

            // boundary of vertex is empty, need to do something in positive dimension only
            if (d > 0) {
                if (n_threads > 1) {
                    tf::Executor executor(n_threads);
                    tf::Taskflow taskflow;
                    taskflow.for_each_index((size_t)0, size_in_dimension(d), (size_t)1,
                            [this, d, missing_ok, &result](size_t col_idx) {
                                auto& sigma = cells_[col_idx + dim_first(d)];
                                auto& col = result[col_idx];
                                col.reserve(d + 1);

                                for(const auto& tau_vertices: sigma.boundary()) {
                                    if (missing_ok) {
                                        auto iter = uid_to_sorted_id.find(tau_vertices);
                                        if (iter != uid_to_sorted_id.end()) {
                                            col.push_back(iter->second);
                                        }
                                    } else {
                                        col.push_back(uid_to_sorted_id.at(tau_vertices));
                                    }
                                }
                                std::sort(col.begin(), col.end());
                            });
                    executor.run(taskflow).get();
                } else {
                    for(size_t col_idx = 0; col_idx < size_in_dimension(d); ++col_idx) {
                        auto& sigma = cells_[col_idx + dim_first(d)];
                        auto& col = result[col_idx];
                        col.reserve(d + 1);

                        for(const auto& tau_vertices: sigma.boundary()) {
                            if (missing_ok) {
                                auto iter = uid_to_sorted_id.find(tau_vertices);
                                if (iter != uid_to_sorted_id.end()) {
                                    col.push_back(iter->second);
                                }
                            } else {
                                col.push_back(uid_to_sorted_id.at(tau_vertices));
                            }
                        }

                        std::sort(col.begin(), col.end());
                    }
                }
            }
            return result;
        }

        BoundaryMatrix boundary_matrix_rel(const typename Cell::UidSet& relative) const
        {
            CALI_CXX_MARK_FUNCTION;

            BoundaryMatrix result;
            result.reserve(size());

            for(dim_type d = 0; d <= max_dim(); ++d) {
                auto m = boundary_matrix_in_dimension_rel(d, relative);
                result.insert(result.end(), std::make_move_iterator(m.begin()), std::make_move_iterator(m.end()));
            }

            return result;
        }

        BoundaryMatrix boundary_matrix_in_dimension_rel(dim_type d, const typename Cell::UidSet& relative) const
        {
            CALI_CXX_MARK_FUNCTION;
            BoundaryMatrix result(size_in_dimension(d));
            // fill D with empty vectors

            // boundary of vertex is empty, need to do something in positive dimension only
            if (d > 0)
                for(size_t col_idx = 0; col_idx < size_in_dimension(d); ++col_idx) {

                    const auto& sigma = cells_[col_idx + dim_first(d)];

                    if (relative.find(sigma.get_uid()) != relative.end())
                        continue;

                    auto& col = result[col_idx];
                    col.reserve(d + 1);

                    for(const auto& tau_vertices: sigma.boundary()) {
                        if (relative.find(tau_vertices) == relative.end())
                            col.push_back(uid_to_sorted_id.at(tau_vertices));
                    }

#ifdef OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                    if (col_idx % 100 == 0) {
                      OINEUS_CHECK_FOR_PYTHON_INTERRUPT
                    }
#endif

                    std::sort(col.begin(), col.end());

#ifndef NDEBUG
                    // verify that there are no duplicates
                    std::set<Int> col_check {col.begin(), col.end()};
                    if (col_check.size() != col.size())
                        throw std::runtime_error("Bad uid computation");
#endif

                }

            return result;
        }

        BoundaryMatrix coboundary_matrix(int n_threads=1) const
        {
            CALI_CXX_MARK_FUNCTION;
            return antitranspose(boundary_matrix(n_threads), size());
        }

        template<typename I, typename R, size_t D>
        friend class Grid;

        dim_type dim_by_id(Int id) const
        {
            for(dim_type dim = 0; dim <= max_dim(); ++dim) {
                if (static_cast<Int>(dim_first(dim)) <= id and static_cast<Int>(dim_last(dim)) >= id)
                    return dim;
            }
            throw std::runtime_error("Error in dim_by_id");
        }

        // ranges of id and sorted id are the same, since dimension is preserved in sorting
        dim_type dim_by_sorted_id(Int sorted_id) const { return dim_by_id(sorted_id); }

        Real value_by_sorted_id(Int sorted_id) const
        {
            return cells_.at(sorted_id).get_value();
        }

        Real value_by_uid(const CellUid& vs) const
        {
            return cells_.at(uid_to_sorted_id.at(vs)).get_value();
        }

        auto get_sorted_id_by_uid(const CellUid& uid) const
        {
            return uid_to_sorted_id.at(uid);
        }

        Cell get_cell_by_uid(const CellUid& uid) const
        {
            return cells_[uid_to_sorted_id.at(uid)];
        }

        bool contains_cell_with_uid(const CellUid& uid) const
        {
            return uid_to_sorted_id.count(uid) == 1;
        }

        Real min_value() const
        {
            if (cells_.empty())
                return negate_ ? std::numeric_limits<Real>::max() : -std::numeric_limits<Real>::max();
            else
                return cells_[0].get_value();
        }

        Real min_value(dim_type d) const
        {
            return cells_.at(dim_first(d)).get_value();
        }

        const CellVector& cells() const { return cells_; }
        CellVector& cells() { return cells_; }
        CellVector cells_copy() const { return cells_; }

        Cell get_cell(size_t i)  const { return cells_.at(i); }
        Cell get_simplex(size_t i) const { return cells_.at(i); }

        auto get_sorted_id(int i) const
        {
            return id_to_sorted_id_.at(i);
        }

        auto get_id_by_sorted_id(int sorted_id) const
        {
            return sorted_id_to_id_.at(sorted_id);
        }

        auto get_sorting_permutation() const
        {
            return id_to_sorted_id_;
        }

        auto get_inv_sorting_permutation() const
        {
            return sorted_id_to_id_;
        }

        bool negate() const { return negate_; }

        bool cmp(Real a, Real b) const { return negate() ? (a > b) : (a < b); }

        Real infinity() const
        {
            static_assert(std::numeric_limits<Real>::has_infinity, "Real does not have inf");
            return negate() ? -std::numeric_limits<Real>::infinity() : std::numeric_limits<Real>::infinity();
        }

        // without cohomology, index in D == index in filtration. For cohomology, indexation is reversed
        // these two functions convert
        [[nodiscard]] size_t index_in_matrix(size_t cell_idx, bool dualize) const { return dualize ? size() - cell_idx - 1 : cell_idx; }
        [[nodiscard]] size_t index_in_filtration(size_t matrix_idx, bool dualize) const { return dualize ? size() - matrix_idx - 1 : matrix_idx; }

        void set_values(const std::vector<Real>& new_values, int n_threads=1)
        {
            if (new_values.size() != cells_.size())
                throw std::runtime_error("new_values.size() != cells_.size()");

            for(size_t i = 0; i < new_values.size(); ++i)
                cells_[i].value_ = new_values[i];

            sort_and_set(n_threads, false);
        }

        // return true, if it's a subfiltration (subset) of a true filtration
        // if so, the
        bool is_subfiltration() const { return is_subfiltration_; }

        template<class P>
        Filtration subfiltration(const P& pred)
        {
            Filtration result;

            result.is_subfiltration_ = true;

            std::set<dim_type> dims;

            Int sorted_id = 0;
            for(const auto& cell : cells_) {
                if (pred(cell)) {
                    dims.insert(cell.dim());
                    result.cells_.push_back(cell);
                    result.uid_to_sorted_id[cell.get_uid()] = sorted_id;
                    result.id_to_sorted_id_[cell.get_id()] = sorted_id;
                    result.sorted_id_to_id_.push_back(cell.get_id());
                    result.cells_.back().sorted_id_ = sorted_id;
                    sorted_id++;
                }
            }

            if (result.size() == 0)
                return Filtration();

            dim_type max_dim = result.cells_.back().dim();

            result.set_dim_info(max_dim, dims);

            return result;
        }

        auto begin() { return cells_.begin(); }
        auto end()   { return cells_.end(); }

    private:
        // data
        bool negate_;
        CellVector cells_;
        bool is_subfiltration_ {false};

        std::unordered_map<CellUid, Int, UidHasher> uid_to_sorted_id;
        std::vector<Int> id_to_sorted_id_;
        std::vector<Int> sorted_id_to_id_;

        std::vector<Int> dim_first_;
        std::vector<Int> dim_last_;

        void set_dim_info()
        {
            dim_type curr_dim = 0;
            dim_first_.push_back(0);
            for(size_t i = 0; i < size(); ++i)
                if (cells_[i].dim() != curr_dim) {
                    if (cells_[i].dim() != curr_dim + 1)
                        throw std::runtime_error("Wrong dimension");
                    assert(i >= 1 and cells_[i].dim() == curr_dim + 1);
                    dim_last_.push_back(static_cast<Int>(i) - 1);
                    dim_first_.push_back(i);
                    curr_dim = cells_[i].dim();
                }
            dim_last_.push_back(size() - 1);
        }

        // for subfiltration case only:
        // some dimensions might be missing, present_dims contains the dimensions
        // for which there are cells in subfiltration
        void set_dim_info(dim_type max_dim, const std::set<dim_type>& present_dims)
        {
            dim_first_ = std::vector<Int>(max_dim + 1, 0);
            dim_last_ = std::vector<Int>(max_dim + 1, 0);

            // for missing dimensions set dim_last to be smaller
            // than dim_first so that we skip it when looping over
            // dimensions
            for(dim_type d = 0; d < max_dim; ++d) {
                if (present_dims.find(d) == present_dims.end()) {
                    dim_last_[d] = -1;
                }
            }

            if (cells_.empty()) {
                return;
            }

            dim_type curr_dim = cells_[0].dim();
            dim_first_[curr_dim] = 0;
            for(size_t i = 0; i < size(); ++i) {
                dim_type new_dim = cells_[i].dim();
                if (new_dim != curr_dim) {
                    dim_last_[curr_dim] = i - 1;
                    dim_first_[new_dim] = i;
                    curr_dim = new_dim;
                }
            }

            dim_last_[curr_dim] = size() - 1;
        }

        // sort cells and assign sorted_ids
        void sort_dim_only()
        {
            CALI_CXX_MARK_FUNCTION;

            std::map<dim_type, CellVector> dim_to_cells;

            for(auto cell : cells_) {
                dim_to_cells[cell.dim()].push_back(cell);
            }

            uid_to_sorted_id.clear();
            uid_to_sorted_id.reserve(cells_.size());

            size_t sorted_id = 0;

            cells_.clear();

            for(auto& [dim, cells_in_dim] : dim_to_cells) {
                for(auto& cell : cells_in_dim) {
                    id_to_sorted_id_[cell.get_id()] = sorted_id;
                    sorted_id_to_id_.at(sorted_id) = cell.get_id();
                    cell.sorted_id_ = sorted_id;
                    uid_to_sorted_id[cell.get_uid()] = sorted_id;

                    cells_.push_back(cell);

                    sorted_id++;
                }
            }
        }

        // sort cells and assign sorted_ids
        void sort_and_set(int n_threads, bool set_uid)
        {
            CALI_CXX_MARK_FUNCTION;

            uid_to_sorted_id.clear();
            uid_to_sorted_id.reserve(cells_.size());

            auto cmp = [this](const Cell& sigma, const Cell& tau)
                {
                      Real v_sigma = negate() ? -sigma.get_value() : sigma.get_value();
                      Real v_tau = negate() ? -tau.get_value() : tau.get_value();
                      dim_type d_sigma = sigma.dim(), d_tau = tau.dim();
                      auto sigma_id = sigma.get_id(), tau_id = tau.get_id();
                      return std::tie(d_sigma, v_sigma, sigma_id) < std::tie(d_tau, v_tau, tau_id);
                };

            CALI_MARK_BEGIN("TF_SORT_AND_SET");
            tf::Executor executor(n_threads);
            tf::Taskflow taskflow;
            tf::Task set_id_and_uid = taskflow.for_each_index(static_cast<Int>(0), static_cast<Int>(size()), static_cast<Int>(1),
                    [this, set_uid](Int sorted_id){
                    auto& sigma = this->cells_[sorted_id];
                    if (set_uid) sigma.set_uid();
                    sigma.set_id(sorted_id);
                    });
            tf::Task sort_task = taskflow.sort(cells_.begin(), cells_.end(), cmp);
            tf::Task post_sort = taskflow.for_each_index(static_cast<Int>(0), static_cast<Int>(size()), static_cast<Int>(1),
                    [this](Int sorted_id){
                    auto& sigma = this->cells_[sorted_id];
                    this->id_to_sorted_id_[sigma.get_id()] = sorted_id;
                    this->sorted_id_to_id_[sorted_id] = sigma.get_id();
                    sigma.sorted_id_ = sorted_id;
                    });
            sort_task.succeed(set_id_and_uid);
            post_sort.succeed(sort_task);
            executor.run(taskflow).get();
            CALI_MARK_END("TF_SORT_AND_SET");

            // unordered_set is not thread-safe, cannot do in parallel
            CALI_MARK_BEGIN("filtration_init_housekeeping-uid");
            for(size_t sorted_id = 0; sorted_id < size(); ++sorted_id) {
                auto& sigma = cells_[sorted_id];
                uid_to_sorted_id[sigma.get_uid()] = sorted_id;
            }
            CALI_MARK_END("filtration_init_housekeeping-uid");
        }

        template<typename C, typename R>
        friend std::ostream& operator<<(std::ostream&, const Filtration<C, R>&);
    };

    template<typename C, typename R>
    std::ostream& operator<<(std::ostream& out, const Filtration<C, R>& fil)
    {
        out << "Filtration(size = " << fil.size() << ", " << "\ncells = [";
        dim_type d = 0;
        for(const auto& sigma : fil.cells()) {
            if (sigma.dim() == d) {
                out << "\n# Dimension: " << d++ << "\n";
            }
            out << sigma << ",\n";
        }
        if (fil.size() == 0)
            out << "\n";
        out << ", dim_first = [";
        for(auto x : fil.dim_first())
            out << x << ",";
        out << "]\n";
        out << ", dim_last = [";
        for(auto x : fil.dim_last())
            out << x << ",";
        out << "]\n";
        out << ");";
        return out;
    }

    template<class C, class R>
    Filtration<C, R> min_filtration(const Filtration<C, R>& fil_1, const Filtration<C, R>& fil_2)
    {
        if (fil_1.negate() != fil_2.negate())
            throw std::runtime_error("Cannot construct min filtration from two filtrations with opposite order");

        if (fil_1.size() != fil_2.size())
            throw std::runtime_error("Refuse to construct min filtration from two filtrations of different sizes");

        auto cells = fil_1.cells_copy();

        for(CellWithValue<C, R>& cell : cells) {
            auto cell_index_2 = fil_2.get_sorted_id_by_uid(cell.get_uid());
            R value_2 = fil_2.get_cell_value(cell_index_2);
            if (fil_1.cmp(value_2, cell.get_value())) {
                cell.set_value(value_2);
            }
        }

        // retain ids from fil_1
        return Filtration<C, R>(cells, fil_1.negate(), 1, false, false);
    }


    template<class C, class R>
    std::tuple<Filtration<C, R>, std::vector<size_t>, std::vector<size_t>> min_filtration_with_indices(const Filtration<C, R>& fil_1, const Filtration<C, R>& fil_2)
    {
        using Fil = Filtration<C, R>;
        using Cell = typename Fil::Cell;
        using Real = R;

        assert(fil_1.size() == fil_2.size());
        assert(fil_1.negate() == fil_2.negate());

        bool negate = fil_1.negate();

        const std::vector<Cell>& cells = fil_1.cells();

        std::vector<std::tuple<Real, dim_type, size_t, size_t>> to_sort;

        to_sort.reserve(fil_1.size());

        for(size_t idx_1 = 0; idx_1 < fil_1.size(); ++idx_1) {
            size_t idx_2 = fil_2.get_sorted_id_by_uid(cells[idx_1].get_uid());

            Real value_1 = fil_1.get_cell_value(idx_1);
            Real value_2 = fil_2.get_cell_value(idx_2);

            Real min_value = negate ? std::min(-value_1, -value_2) : std::min(value_1, value_2);
            to_sort.emplace_back(min_value, cells[idx_1].dim(), idx_1, idx_2);
        }

        // lexicographic comparison: first by value, then by dimension
        std::sort(to_sort.begin(), to_sort.end());

        std::vector<Cell> min_cells;

        min_cells.reserve(fil_1.size());

        std::vector<size_t> perm_1, perm_2;

        for(const auto& [value, dim, index_1, index_2] : to_sort) {
            min_cells.emplace_back(cells[index_1]);
            min_cells.back().set_value(negate ? -value : value);

            perm_1.push_back(index_1);
            perm_2.push_back(index_2);
        }

        Filtration<C, R> new_fil(std::move(min_cells), negate);

        return { new_fil, perm_1, perm_2 };
    }


} // namespace oineus
