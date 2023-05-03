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

    template<typename Int_, typename Real_, typename L_>
    class Filtration {
    public:
        using Int = Int_;
        using Real = Real_;
        using ValueLocation = L_;

        using FiltrationSimplex = Simplex<Int, Real, ValueLocation>;
        using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
        using IntVector = std::vector<Int>;
        using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;

        Filtration() = default;

        Filtration(const FiltrationSimplexVector& _simplices, bool _negate, int n_threads = 1)
                :
                negate_(_negate),
                simplices_(_simplices)
        {
            Timer timer;
            timer.reset();

            set_ids();
            sort(n_threads);
            set_dim_info();
            assert(std::all_of(simplices_.begin(), simplices_.end(),
                    [](const FiltrationSimplex& sigma) { return sigma.is_valid_filtration_simplex(); }));

//            std::cerr << "Filtration ctor done, elapsed: " << timer.elapsed() << std::endl;
        }

        size_t size() const { return simplices_.size(); }

        size_t dim_first(dim_type d) const { return dim_first_.at(d); }
        size_t dim_last(dim_type d) const { return dim_last_.at(d); }

        auto dim_first() const { return dim_first_; }
        auto dim_last() const { return dim_last_; }

        Real get_simplex_value(size_t i) { return simplices_[i].value(); }
        ValueLocation cvl(size_t simplex_idx) const { return simplices()[simplex_idx].critical_value_location_; }

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
                    auto& sigma = simplices_[col_idx + dim_first(d)];
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
            if (simplices_.empty())
                return negate_ ? std::numeric_limits<Real>::max() : -std::numeric_limits<Real>::max();
            else
                return simplices_[0].value();
        }

        Real min_value(dim_type d) const
        {
            return simplices_.at(dim_first(d)).value();
        }

        const FiltrationSimplexVector& simplices() const { return simplices_; }
        FiltrationSimplexVector& simplices() { return simplices_; }
        FiltrationSimplexVector simplices_copy() const { return simplices_; }

        int get_sorted_id(int i)
        {
            return id_to_sorted_id_[i];
        }

//        decltype(auto) vertices_to_sorted_id() const { return vertices_to_sorted_id_; }
//        decltype(auto) id_to_sorted_id() const { return id_to_sorted_id_; }
//        decltype(auto) sorted_id_to_value() const { return sorted_id_to_value_; }

        bool negate() const { return negate_; }

        bool cmp(Real a, Real b) const { return negate() ? (a > b) : (a < b); }

        Real infinity() const
        {
            static_assert(std::numeric_limits<Real>::has_infinity, "Real does not have inf");
            return negate() ? -std::numeric_limits<Real>::infinity() : std::numeric_limits<Real>::infinity();
        }

        // without cohomology, index in D == index in filtration. For cohomology, indexation is reversed
        // these two functions convert
        [[nodiscard]] size_t index_in_matrix(size_t simplex_idx, bool dualize) const { return dualize ? size() - simplex_idx - 1 : simplex_idx; }
        [[nodiscard]] size_t index_in_filtration(size_t matrix_idx, bool dualize) const { return dualize ? size() - matrix_idx - 1 : matrix_idx; }

    private:
        bool negate_;
        FiltrationSimplexVector simplices_;

        std::map<IntVector, Int> vertices_to_sorted_id_;
        std::vector<Int> id_to_sorted_id_;

        std::vector<Real> sorted_id_to_value_;

        std::vector<size_t> dim_first_;
        std::vector<size_t> dim_last_;

        void set_ids()
        {
            // all vertices have ids already, 0..#vertices-1
            // set ids only on higher-dimensional simplices
            for(size_t id = 0; id < simplices_.size(); ++id) {

                auto& sigma = simplices_[id];

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
                if (simplices_[i].dim() != curr_dim) {
                    if (simplices_[i].dim() != curr_dim + 1)
                        throw std::runtime_error("Wrong dimension");
                    assert(i >= 1 and simplices_[i].dim() == curr_dim + 1);
                    dim_last_.push_back(i - 1);
                    dim_first_.push_back(i);
                    curr_dim = simplices_[i].dim();
                }
            dim_last_.push_back(size() - 1);
        }

        // sort simplices and assign sorted_ids
        void sort([[maybe_unused]] int n_threads = 1)
        {
            Timer timer;

            timer.reset();

            id_to_sorted_id_ = std::vector<Int>(size(), Int(-1));
            vertices_to_sorted_id_.clear();
            sorted_id_to_value_ = std::vector<Real>(size(), std::numeric_limits<Real>::max());

            // sort by dimension first, then by value, then by id
            auto cmp = [this](const FiltrationSimplex& sigma, const FiltrationSimplex& tau) {
              Real v_sigma = this->negate_ ? -sigma.value() : sigma.value();
              Real v_tau = this->negate_ ? -tau.value() : tau.value();
              Int d_sigma = sigma.dim(), d_tau = tau.dim();
              return std::tie(d_sigma, v_sigma, sigma.id_) < std::tie(d_tau, v_tau, tau.id_);
            };

            tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, n_threads);
            tbb::parallel_sort(simplices_, cmp);

            for(size_t sorted_id = 0; sorted_id < size(); ++sorted_id) {
                auto& sigma = simplices_[sorted_id];

                id_to_sorted_id_[sigma.id_] = sorted_id;
                sigma.sorted_id_ = sorted_id;
                vertices_to_sorted_id_[sigma.vertices_] = sorted_id;
                sorted_id_to_value_[sorted_id] = sigma.value();
            }
        }

        template<typename I, typename R, typename L>
        friend std::ostream& operator<<(std::ostream&, const Filtration<I, R, L>&);
    };

    template<typename I, typename R, typename L>
    std::ostream& operator<<(std::ostream& out, const Filtration<I, R, L>& fil)
    {
        out << "Filtration(size = " << fil.size() << ")[" << "\n";
        dim_type d = 0;
        for(size_t idx = 0; idx < fil.size(); ++idx) {
            if (idx == fil.dim_last(d))
                d++;
            if (idx == fil.dim_first(d))
                out << "Dimension: " << d << "\n";
            out << fil.simplices()[idx] << "\n";
        }
        out << "]";
        return out;
    }

} // namespace oineus
