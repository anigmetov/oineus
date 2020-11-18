#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <ostream>

#include "simplex.h"
#include "matrix.h"

namespace oineus {

    template<typename Int_, typename Real_, size_t D>
    class Grid;

    template<typename Int_, typename Real_>
    class Filtration {
    public:
        using Int = Int_;
        using Real = Real_;

        using FiltrationSimplex = Simplex<Int_, Real_>;
        using FiltrationSimplexVector = std::vector<FiltrationSimplex>;
        using IntVector = std::vector<Int>;
        using BoundaryMatrix = SparseMatrix<Int>;

        Filtration() = default;

        Filtration(std::vector<FiltrationSimplexVector>&& _dim_to_simplices, bool _negate) :
            dim_to_simplices_(_dim_to_simplices),
            negate_(_negate)
        {
            set_ids();
            sort(_negate);

            for(Int d = 0; d < dim_to_simplices_.size(); ++d) {
                assert(std::all_of(dim_to_simplices_.at(d).begin(), dim_to_simplices_.at(d).end(),
                            [](const FiltrationSimplex& sigma) { return sigma.is_valid_filtration_simplex(); }));
            }
        }

        Int size_in_dimension(Int d) const
        {
            if (d < 0)
                throw std::runtime_error("Negative dimension");

            if (d >= dim_to_simplices_.size())
                return 0;

            return static_cast<Int>(dim_to_simplices_[d].size());
        }

        Int size() const
        {
            return std::accumulate(dim_to_simplices_.cbegin(), dim_to_simplices_.cend(), Int(0),
                    [](Int a, const FiltrationSimplexVector& v) { return static_cast<Int>(a + v.size()); });
        }

        BoundaryMatrix boundary_matrix_full() const
        {
            BoundaryMatrix result;
            result.data.reserve(size());

            for(Int d = 0; d < dim_to_simplices_.size(); ++d) {
                result.append(boundary_matrix_in_dimension(d));
            }

            return result;
        }

        BoundaryMatrix boundary_matrix_in_dimension(Int d) const
        {
            const auto& simplices = dim_to_simplices_[d];

            BoundaryMatrix result;
            result.data.reserve(simplices.size());

            for(const auto& sigma : simplices) {
                typename BoundaryMatrix::IntSparseColumn col;
                if (d != 0) {
                    col.reserve(d + 1);

                    for(const auto& tau_vertices : sigma.boundary())
                        col.push_back(vertices_to_sorted_id_.at(tau_vertices));

                    std::sort(col.begin(), col.end());
                } // else: boundary of vertex is empty
                result.data.push_back(col);
            }

            return result;
        }

        template<typename I, typename R, size_t D>
        friend class Grid;

        Int dim_by_sorted_id(Int sorted_id) const
        {
            return sorted_id_to_dimension_[sorted_id];
        }

        Real value_by_sorted_id(Int sorted_id) const
        {
            return sorted_id_to_value_[sorted_id];
        }

    private:
        std::vector<FiltrationSimplexVector> dim_to_simplices_;

        bool negate_;

        std::map<IntVector, Int> vertices_to_sorted_id_;
        std::vector<Int> id_to_sorted_id_;

        std::vector<Int> sorted_id_to_dimension_;
        std::vector<Real> sorted_id_to_value_;

        void set_ids()
        {
            // all vertices have ids already, 0..#vertices-1
            // set ids only on higher-dimensional simplices
            Int id = dim_to_simplices_.at(0).size();
            for(size_t d = 1; d < dim_to_simplices_.size(); ++d)
                for(auto& sigma : dim_to_simplices_[d])
                    sigma.id_ = id++;
        }

        // sort simplices and assign sorted_ids
        void sort(bool negate)
        {
            id_to_sorted_id_ = std::vector<Int>(size(), Int(-1));
            vertices_to_sorted_id_.clear();


            sorted_id_to_dimension_ = std::vector<Int>(size(), Int(-1));
            sorted_id_to_value_ = std::vector<Real>(size(), std::numeric_limits<Real>::max());

            // ignore ties
            auto cmp = [negate](const FiltrationSimplex& sigma, const FiltrationSimplex& tau)
                                {
                                    if (negate)
                                        return sigma.value_ > tau.value_;
                                    else
                                        return sigma.value_ < tau.value_;
                                };

            Int s_id_shift = 0;
            for(size_t d = 0; d < dim_to_simplices_.size(); ++d) {

                auto& simplices = dim_to_simplices_[d];

                std::sort(simplices.begin(), simplices.end(), cmp);

                for(Int s_id = 0; s_id < simplices.size(); ++s_id) {

                    auto id = simplices[s_id].id_;
                    auto sorted_id = s_id_shift + s_id;
                    auto& sigma = simplices[s_id];

                    id_to_sorted_id_[id] = sorted_id;
                    sigma.sorted_id_ = sorted_id;
                    vertices_to_sorted_id_[sigma.vertices_] = sorted_id;
                    sorted_id_to_dimension_[sorted_id] = sigma.dim();
                    sorted_id_to_value_[sorted_id] = sigma.value();
                }

                s_id_shift += simplices.size();
            }
        }

        template<typename I, typename R>
        friend std::ostream& operator<<(std::ostream&, const Filtration<I, R>&);
    };


    template<typename I, typename R>
    std::ostream& operator<<(std::ostream& out, const Filtration<I, R>& fil)
    {
        out << "Filtration(size = " << fil.size() <<  "[" << "\n";
        for(size_t d = 0; d < fil.dim_to_simplices_.size(); ++d) {
            out << "Dimension: " << d << "\n";

            for(const auto& sigma : fil.dim_to_simplices_[d])
                out << sigma << "\n";
        }
        out << "]";
        return out;
    }

} // namespace oineus
