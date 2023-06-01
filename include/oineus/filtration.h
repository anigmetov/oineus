#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <ostream>

#include <algorithm>

#include <tbb/parallel_sort.h>
#include <tbb/global_control.h>

#include <icecream/icecream.hpp>

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
    template<typename Cell, typename L>
    struct FiltrationCritValLocs {
        Filtration<Cell> filtration;
        std::vector<L> critical_value_locations;
    };


    template<class Cell_>
    class Filtration {
    public:
        using Cell = Cell_;
        using Int = typename Cell::Int;
        using Real = typename Cell::Real;
        using CellVector = std::vector<Cell>;
        using IntVector = std::vector<Int>;
        using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;

        Filtration() = default;

        // use move constructor to move cells
        Filtration(CellVector&& _simplices, bool _negate, int n_threads = 1)
                :
                negate_(_negate),
                cells_(_simplices)
        {
            CALI_CXX_MARK_FUNCTION;

            set_ids();
            sort(n_threads);
            set_dim_info();
            assert(std::all_of(cells_.begin(), cells_.end(),
                    [](const Cell& sigma) { return sigma.is_valid_filtration_simplex(); }));
        }

        // copy cells
        Filtration(const CellVector& cells, bool _negate, int n_threads = 1)
                :
                negate_(_negate),
                cells_(cells)
        {
            CALI_CXX_MARK_FUNCTION;

            set_ids();
            sort(n_threads);
            set_dim_info();
            assert(std::all_of(cells_.begin(), cells_.end(),
                    [](const Cell& sigma) { return sigma.is_valid_filtration_simplex(); }));

        }

        size_t size() const { return cells_.size(); }

        size_t dim_first(dim_type d) const { return dim_first_.at(d); }
        size_t dim_last(dim_type d) const { return dim_last_.at(d); }

        auto dim_first() const { return dim_first_; }
        auto dim_last() const { return dim_last_; }

        Real get_cell_value(size_t i) { return cells_[i].value(); }

        size_t size_in_dimension(dim_type d) const
        {
            if (d > max_dim())
                return 0;
            Int result = dim_last(d) - dim_first(d) + 1;
            if (result < 0)
                throw std::runtime_error("dim_last less than dim_first");
            return static_cast<size_t>(result);
        }

        dim_type max_dim() const { return dim_last_.size() - 1; }

        BoundaryMatrix boundary_matrix_full() const
        {
            CALI_CXX_MARK_FUNCTION;

            BoundaryMatrix result;
            result.reserve(size());

            for(dim_type d = 0; d <= max_dim(); ++d) {
                auto m = boundary_matrix_in_dimension(d);
                result.insert(result.end(), std::make_move_iterator(m.begin()), std::make_move_iterator(m.end()));
            }

            return result;
        }

        BoundaryMatrix boundary_matrix_in_dimension(dim_type d) const
        {
            BoundaryMatrix result(size_in_dimension(d));
            // fill D with empty vectors

            // boundary of vertex is empty, need to do something in positive dimension only
            if (d > 0)
                for(size_t col_idx = 0; col_idx < size_in_dimension(d); ++col_idx) {
                    auto& sigma = cells_[col_idx + dim_first(d)];
                    auto& col = result[col_idx];
                    col.reserve(d + 1);

                    for(const auto& tau_vertices: sigma.boundary()) {
                        col.push_back(vertices_to_sorted_id_.at(tau_vertices));
                    }

                    std::sort(col.begin(), col.end());
                }

            return result;
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
            return sorted_id_to_value_[sorted_id];
        }

        Real min_value() const
        {
            if (cells_.empty())
                return negate_ ? std::numeric_limits<Real>::max() : -std::numeric_limits<Real>::max();
            else
                return cells_[0].value();
        }

        Real min_value(dim_type d) const
        {
            return cells_.at(dim_first(d)).value();
        }

        const CellVector& cells() const { return cells_; }
        CellVector& cells() { return cells_; }
        CellVector cells_copy() const { return cells_; }

        int get_sorted_id(int i)
        {
            return id_to_sorted_id_[i];
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

        void update(const std::vector<Real>& new_values, int n_threads=1)
        {
            if (new_values.size() != cells_.size())
                throw std::runtime_error("new_values.size() != cells_.size()");

            for(size_t i = 0; i < new_values.size(); ++i)
                cells_[i].value_ = new_values[i];

            sort(n_threads);
        }

    private:
        // data
        bool negate_;
        CellVector cells_;

        std::map<IntVector, Int> vertices_to_sorted_id_;
        std::vector<Int> id_to_sorted_id_;

        std::vector<Real> sorted_id_to_value_;

        std::vector<size_t> dim_first_;
        std::vector<size_t> dim_last_;

        // private methods
        void set_ids()
        {
            // all vertices have ids already, 0..#vertices-1
            // set ids only on higher-dimensional cells
            for(size_t id = 0; id < cells_.size(); ++id) {

                auto& sigma = cells_[id];

                if (sigma.dim() == 0 and sigma.id_ != static_cast<Int>(id))
                    throw std::runtime_error("Vertex id and order of vertices do not match");
                else
                    sigma.id_ = static_cast<Int>(id);
            }
        }

        void set_dim_info()
        {
            Int curr_dim = 0;
            dim_first_.push_back(0);
            for(size_t i = 0; i < size(); ++i)
                if (cells_[i].dim() != curr_dim) {
                    if (cells_[i].dim() != curr_dim + 1)
                        throw std::runtime_error("Wrong dimension");
                    assert(i >= 1 and cells_[i].dim() == curr_dim + 1);
                    dim_last_.push_back(i - 1);
                    dim_first_.push_back(i);
                    curr_dim = cells_[i].dim();
                }
            dim_last_.push_back(size() - 1);
        }

        // sort cells and assign sorted_ids
        void sort([[maybe_unused]] int n_threads = 1)
        {
            CALI_CXX_MARK_FUNCTION;

            id_to_sorted_id_ = std::vector<Int>(size(), Int(-1));
            vertices_to_sorted_id_.clear();
            sorted_id_to_value_ = std::vector<Real>(size(), std::numeric_limits<Real>::max());

            // sort by dimension first, then by value, then by id
            auto cmp = [this](const Cell& sigma, const Cell& tau) {
              Real v_sigma = this->negate_ ? -sigma.value() : sigma.value();
              Real v_tau = this->negate_ ? -tau.value() : tau.value();
              Int d_sigma = sigma.dim(), d_tau = tau.dim();
              return std::tie(d_sigma, v_sigma, sigma.id_) < std::tie(d_tau, v_tau, tau.id_);
            };

            if (n_threads > 1) {
                tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, n_threads);
                tbb::parallel_sort(cells_, cmp);
            } else {
                std::sort(cells_.begin(), cells_.end(), cmp);
            }

            for(size_t sorted_id = 0; sorted_id < size(); ++sorted_id) {
                auto& sigma = cells_[sorted_id];

                id_to_sorted_id_[sigma.id_] = sorted_id;
                sigma.sorted_id_ = sorted_id;
                vertices_to_sorted_id_[sigma.vertices_] = sorted_id;
                sorted_id_to_value_[sorted_id] = sigma.value();
            }
        }

        template<typename C>
        friend std::ostream& operator<<(std::ostream&, const Filtration<C>&);
    };

    template<typename C>
    std::ostream& operator<<(std::ostream& out, const Filtration<C>& fil)
    {
        out << "Filtration(size = " << fil.size() << ")[" << "\n";
        dim_type d = 0;
        for(size_t idx = 0; idx < fil.size(); ++idx) {
            if (idx == fil.dim_last(d))
                d++;
            if (idx == fil.dim_first(d))
                out << "Dimension: " << d << "\n";
            out << fil.cells()[idx] << "\n";
        }
        out << "]";
        return out;
    }

} // namespace oineus
