#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <ostream>

#include <algorithm>

#include "timer.h"
#include "simplex.h"
#include "filtration.h"
#include "diagram.h"
#include "decomposition.h"
#include "params.h"
#include "product_cell.h"

namespace oineus {

template<class Cell_>
class CylinderFiltration {
public:
    using Cell = ProductCell<Cell_, Cell_>;
    using Fil = Filtration<Cell_>;
    using Int = typename Cell_::Int;
    using Real = typename Cell_::Real;
    using Mapping = std::vector<size_t>; // for each simplex of Fil_1, contains its index in Fil_2
    using BoundaryMatrix = typename VRUDecomposition<Int>::MatrixData;

    CylinderFiltration() = default;

    CylinderFiltration(const Fil& fil_domain,
            const Fil& fil_codomain,
            const Mapping& map)
            :
            fil_domain_(fil_domain),
            fil_codomain_(fil_codomain)
    {

        std::vector<Cell> all_cells;

        Simplex<Int, Real> pt;
        Cell_ segment;

        for(size_t dom_idx = 0; dom_idx < fil_domain.cells().size(); ++dom_idx) {
            auto dom_cell = fil_domain.cells()[dom_idx];
            auto image_cell = fil_codomain.cells()[map[dom_idx]];
            all_cells.emplace_back(dom_cell, pt, dom_cell.value());
            all_cells.emplace_back(image_cell, pt, image_cell.value())
        }


        check_sizes();
    }

    void check_sizes()
    {
        if (fil_domain_.max_dim() != fil_codomain_.max_dim())
            throw std::runtime_error("not supported");

        for(dim_type d = 0 ; d < fil_domain_.max_dim() ; ++d) {
            if (fil_domain_.size_in_dimension(d) != fil_codomain_.size_in_dimension(d))
                throw std::runtime_error("not supported");
        }
    }

    size_t size() const { return fil_domain_.size(); }

    size_t dim_first(dim_type d) const { return fil_domain_.dim_first(d); }
    size_t dim_last(dim_type d) const { return fil_domain_.dim_last(d); }

    size_t size_in_dimension(dim_type d) const
    {
        if (d > max_dim())
            return 0;
        Int result = dim_last(d) - dim_first(d) + 1;
        if (result < 0)
            throw std::runtime_error("dim_last less than dim_first");
        return static_cast<size_t>(result);
    }

    dim_type max_dim() const { return fil_domain_.max_dim(); }

    size_t row_index(Int i, bool dualize) const { return fil_domain_.index_in_filtration(i, dualize); }
    size_t col_index(Int i, bool dualize) const { return fil_codomain_.index_in_filtration(i, dualize); }

    Real row_value(Int i) const { return fil_domain_.value_by_sorted_id(i); }
    Real col_value(Int i) const { return fil_codomain_.value_by_sorted_id(i); }

    dim_type dim_by_sorted_id(Int i) const { return fil_domain_.dim_by_sorted_id(i); }

    BoundaryMatrix boundary_matrix_full() const
    {
        CALI_CXX_MARK_FUNCTION;

        BoundaryMatrix result;
        result.reserve(size());

        for(dim_type d = 0 ; d <= max_dim() ; ++d) {
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
            for(size_t codomain_idx = 0 ; codomain_idx < size_in_dimension(d) ; ++codomain_idx) {
                auto& sigma = fil_codomain_.cells()[codomain_idx + dim_first(d)];
                auto& col = result[codomain_idx];
                col.reserve(d + 1);

                for(const auto& tau_vertices: sigma.boundary()) {
                    col.push_back(fil_domain_.get_sorted_id_by_vertices(tau_vertices));
                }

                std::sort(col.begin(), col.end());
            }

        return result;
    }

    Real domain_negate() const { return fil_domain_.negate(); }
    bool codomain_negate() const { return fil_codomain_.negate(); }

    template<typename C>
    friend std::ostream& operator<<(std::ostream&, const InclusionFiltration<C>&);
private:
    Fil fil_domain_;
    Fil fil_codomain_;
};

template<typename C>
std::ostream& operator<<(std::ostream& out, const InclusionFiltration<C>& fil)
{
    out << "InclusionFiltration(size = " << fil.size() << ", " << "\ncells = [";
    dim_type d = 0;
    for(const auto& sigma: fil.cells()) {
        if (sigma.dim() == d) {
            out << "\n# Dimension: " << d++ << "\n";
        }
        out << sigma << ",\n";
    }
    if (fil.size() == 0)
        out << "\n";
    out << ", dim_first = [";
    for(auto x: fil.dim_first())
        out << x << ",";
    out << "]\n";
    out << ", dim_last = [";
    for(auto x: fil.dim_last())
        out << x << ",";
    out << "]\n";
    out << ");";
    return out;
}

template<class P>
std::unordered_map<P, P> compose_matchings(const std::unordered_map<P, P>& a_to_b, const std::unordered_map<P, P>& b_to_c)
{
    std::unordered_map<P, P> a_to_c;

    for(const auto& [a, b]: a_to_b) {
        auto b_to_c_iter = b_to_c.find(b);
        if (b_to_c_iter != b_to_c.end())
            a_to_c[a] = b_to_c_iter->second;
    }

    return a_to_c;
}

template<class Cell>
Diagrams<typename Cell::Real> get_image_diagram(const VRUDecomposition<typename Cell::Int>& dcmp, InclusionFiltration<Cell>& fil, bool include_inf_points)
{
    using Real = typename Cell::Real;
    using Int = typename Cell::Int;

    if (not dcmp.is_reduced)
        throw std::runtime_error("Cannot compute diagram from non-reduced matrix, call reduce_parallel");

    Diagrams<Real> result(fil.max_dim());

    std::unordered_set<size_t> unmatched_positive;

    auto& r_data = dcmp.r_data;

    if (include_inf_points) {
        for(size_t col_idx = 0 ; col_idx < r_data.size() ; ++col_idx) {
            auto col = &r_data[col_idx];

            if (is_zero(col)) {
                unmatched_positive.insert(fil.col_index(col_idx, dcmp.dualize()));
                continue;
            }
        }
    }

    for(size_t col_idx = 0 ; col_idx < r_data.size() ; ++col_idx) {
        auto col = &r_data[col_idx];

        if (is_zero(col))
            continue;

        // finite point
        Int birth_idx = fil.row_index(low(col), dcmp.dualize());
        Int death_idx = fil.col_index(col_idx, dcmp.dualize());

        if (include_inf_points)
            unmatched_positive.erase(birth_idx);

        dim_type dim = fil.dim_by_sorted_id(birth_idx);

        Real birth = fil.row_value(birth_idx);
        Real death = fil.col_value(death_idx);

        bool include_point = birth != death;
        if (include_point) {
            if (dcmp.dualize()) {
                result.add_point(dim - 1, death, birth, death_idx, birth_idx);
            } else {
                result.add_point(dim, birth, death, birth_idx, death_idx);
            }
        }
    }

    if (include_inf_points) {
        for(size_t row_idx: unmatched_positive) {
            // finite point
            Int birth_idx = fil.row_index(row_idx, dcmp.dualize());
            Int death_idx = std::numeric_limits<Int>::max();

            dim_type dim = fil.dim_by_sorted_id(birth_idx);

            Real birth = fil.row_value(birth_idx);
            Real death = std::numeric_limits<Real>::infinity();
            if (fil.domain_negate())
                death = -death;

            result.add_point(dim, birth, death, birth_idx, death_idx);
        }
    }

    return result;
}

template<class Cell>
auto get_induced_matching(const Filtration<Cell>& fil_domain, const Filtration<Cell>& fil_codomain, int n_threads = 1)
/**
 *
 * @tparam Cell Class of cells in Filtration, usually Simplex
 * @param fil_domain domain of the inclusion
 * @param fil_codomain codomain of the inclusion
 * @return Partial matching of barcodes of fil_domain and fil_codomain. Each bar is represented as DgmPoint<size_t>
 */
{
    using Int = typename Cell::Int;
    using Real = typename Cell::Real;
    using Bar = DgmPoint<Real>;
    using MatchingInDimension = std::unordered_map<Bar, Bar>;
    using Matching = std::map<dim_type, MatchingInDimension>;
    using Decomposition = VRUDecomposition<Int>;

    Matching induced_matching;

    if (fil_domain.size() != fil_codomain.size())
        throw std::runtime_error("bad sizes");

    std::vector<Int> old_ids_dom(fil_domain.size()), old_ids_cod(fil_codomain.size());
    std::vector<Real> domain_values(fil_domain.size()), codomain_values(fil_codomain.size());

    for(size_t i = 0 ; i < fil_domain.size() ; ++i) {
        // tau = f(sigma)
        const Cell& sigma = fil_domain.cells()[i];
        const Cell& tau = fil_codomain.cells()[i];
        old_ids_dom[i] = sigma.id();
        old_ids_cod[i] = tau.id();
        domain_values[i] = sigma.value();
        codomain_values[i] = tau.value();
    }

    InclusionFiltration<Cell> ifil(fil_domain, fil_codomain);

    bool dualize = false;

    Decomposition dcmp_dom(fil_domain, dualize);
    Decomposition dcmp_cod(fil_codomain, dualize);

    Params params;
    params.n_threads = n_threads;
    params.clearing_opt = true;
    params.compute_v = params.compute_u = false;

    dcmp_dom.reduce(params);
    dcmp_cod.reduce(params);

    Decomposition dcmp_image(ifil.boundary_matrix_full(), fil_domain.size());
    params.clearing_opt = false;

    dcmp_image.reduce(params);

    bool include_infinite_points = true;

    auto image_diagrams = get_image_diagram(dcmp_image, ifil, include_infinite_points);
    auto domain_diagrams = dcmp_dom.diagram(fil_domain, include_infinite_points);
    auto codomain_diagrams = dcmp_cod.diagram(fil_codomain, include_infinite_points);

    for(dim_type d = 0 ; d < fil_domain.max_dim(); ++d) {

        std::unordered_map<Real, std::vector<Bar>> birth_to_image_bars, birth_to_dom_bars, death_to_image_bars, death_to_cod_bars;

        for(auto&& p_image: image_diagrams.get_diagram_in_dimension(d)) {
            birth_to_image_bars[p_image.birth].push_back(p_image);
            death_to_image_bars[p_image.death].push_back(p_image);
        }

        for(auto&& p_domain: domain_diagrams.get_diagram_in_dimension(d)) {
            birth_to_dom_bars[p_domain.birth].push_back(p_domain);
        }

        for(auto&& p_codomain: codomain_diagrams.get_diagram_in_dimension(d)) {
            death_to_cod_bars[p_codomain.death].push_back(p_codomain);
        }

        for(auto& [key, bars] : birth_to_image_bars) { std::sort(bars.begin(), bars.end(), std::greater<Bar>()); }
        for(auto& [key, bars] : death_to_image_bars) { std::sort(bars.begin(), bars.end(), std::greater<Bar>()); }
        for(auto& [key, bars] : birth_to_dom_bars) { std::sort(bars.begin(), bars.end(), std::greater<Bar>()); }
        for(auto& [key, bars] : death_to_cod_bars) { std::sort(bars.begin(), bars.end(), std::greater<Bar>()); }

        MatchingInDimension domain_to_image, image_to_codomain;

        for(auto& [birth, domain_bars]: birth_to_dom_bars) {
            // if there are no image bars with this birth, image_bars will be empty and the for loop will not be executed
            auto& image_bars = birth_to_image_bars[birth];

            // TODO: sort domain_bars, image_bars by persistence
            // for now, assume general position, domain_bars and image_bars have size 1
            for(size_t i = 0 ; i < std::min(domain_bars.size(), image_bars.size()) ; ++i) {
                domain_to_image[domain_bars[i]] = image_bars[i];
            }
        }

        for(auto& [death, image_bars]: death_to_image_bars) {
            auto& codomain_bars = death_to_cod_bars[death];

            for(size_t i = 0 ; i < std::min(image_bars.size(), codomain_bars.size()) ; ++i) {
                image_to_codomain[image_bars[i]] = codomain_bars[i];
            }
        }

        induced_matching[d] = compose_matchings(domain_to_image, image_to_codomain);
    }

    return induced_matching;
}

} // namespace oineus
