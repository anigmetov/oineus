#pragma once
#include <utility>
#include <type_traits>
#include <exception>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>

#include "common_defs.h"
#include "simplex.h"
#include "cell_with_value.h"
#include "filtration.h"

namespace oineus {

template<class CellA, class CellB=CellA>
struct ProductCell {
    using Uid = std::pair<typename CellA::Uid, typename CellB::Uid>;
    using Boundary = std::vector<Uid>;
    using Int = typename CellA::Int;

    static_assert(std::is_integral_v<typename CellA::Int>);

    CellA cell_1_;
    CellB cell_2_;
    Int id_ {0};

    ProductCell() = default;
    ProductCell(const ProductCell&) = default;
    ProductCell(ProductCell&&) = default;
    ProductCell& operator=(const ProductCell&) = default;
    ProductCell& operator=(ProductCell&&) = default;

    ProductCell(const CellA& cell_1, const CellB& cell_2)
            :cell_1_(cell_1), cell_2_(cell_2) { }
    ProductCell(CellA&& cell_1, CellB&& cell_2)
            :cell_1_(cell_1), cell_2_(cell_2) { }

    Uid get_uid() const { return {cell_1_.get_uid(), cell_2_.get_uid()}; }

    Int get_id() const { return id_; }
    void set_id(Int new_id) { id_ = new_id; }

    dim_type dim() const { return cell_1_.dim() + cell_2_.dim(); }

    CellA get_factor_1() const { return cell_1_; }
    CellB get_factor_2() const { return cell_2_; }

    static std::string uid_to_string(const Uid& uid)
    {
        std::stringstream ss;

        ss << "[uid_1 = " << CellA::uid_to_string(uid.first);
        ss << ", uid_2 = " << CellB::uid_to_string(uid.second) << "]";

        return ss.str();
    }

    std::string uid_as_string() const
    {
        return uid_to_string(get_uid());
    }


    Boundary boundary() const
    {
        Boundary result;

        for(auto b_1: cell_1_.boundary()) {
            result.push_back({b_1, cell_2_.get_uid()});
        }

        for(auto b_2: cell_2_.boundary()) {
            result.push_back({cell_1_.get_uid(), b_2});
        }

        return result;
    }

    friend std::ostream& operator<<(std::ostream& out, const ProductCell& s)
    {
        out << "ProductCell(factor_1 = " << s.get_factor_1() << ", factor_2 = " << s.get_factor_2() << ", id = " << s.get_id() << ")";
        return out;
    }

    struct UidHasher {
        std::size_t operator()(const ProductCell::Uid& x) const
        {
            std::size_t seed = 0;
            oineus::hash_combine(seed, typename CellA::UidHasher()(x.first));
            oineus::hash_combine(seed, typename CellB::UidHasher()(x.second));
            return seed;
        }
    };
};

template<class Cell, class Real>
std::vector<size_t> get_inclusion_mapping(const Filtration<Cell, Real>& fil_domain, const Filtration<Cell, Real>& fil_target)
{
    std::vector<size_t> result(fil_domain.size(), k_invalid_index);

    for(size_t sigma_idx = 0 ; sigma_idx < fil_domain.size() ; ++sigma_idx) {
        size_t image_sigma_idx = fil_target.get_sorted_id_by_uid(fil_domain.get_cell(sigma_idx).get_uid());
        result.push_back(image_sigma_idx);
    }

    return result;
}

template<class CWV>
void append_products(const std::vector<CWV>& cells, const Simplex<typename CWV::Int>& sigma,
        std::vector<CellWithValue<ProductCell<typename CWV::Cell, Simplex<typename CWV::Int>>, typename CWV::Real>>& result)
{
    using ProdCell = ProductCell<typename CWV::Cell, Simplex<typename CWV::Int>>;

    for(const auto& cell: cells) {
        result.emplace_back(ProdCell(cell.get_cell(), sigma), cell.get_value());
    }
}

template<class Cell, class Real>
Filtration<ProductCell<Cell, Simplex<typename Cell::Int>>, Real>
multiply_filtration(const Filtration<Cell, Real>& fil, const Simplex<typename Cell::Int>& sigma)
{
    using ProdCWV = CellWithValue<ProductCell<Cell, Simplex<typename Cell::Int>>, Real>;

    std::vector<ProdCWV> result_simplices;
    result_simplices.reserve(fil.size());
    append_products(fil.cells(), sigma, result_simplices);

    return { result_simplices, fil.negate() };
}

template<class Cell, class Real>
Filtration<ProductCell<Cell, Simplex<typename Cell::Int>>, Real>
build_mapping_cylinder(const Filtration<Cell, Real>& fil_domain, const Filtration<Cell, Real>& fil_codomain, const Simplex<typename Cell::Int>& v_domain, const Simplex<typename Cell::Int>& v_codomain)
/**
 *
 * @tparam Cell class of cells in filtrations
 * @tparam Real double or float
 * @param fil_domain filtration of space L
 * @param fil_codomain filtration of space K, L \subset K
 * @param v_domain vertex by which cells of fil_domain are multiplied to get the top of the mapping cylinder
 * @param v_codomain vertex by which cells of fil_domain are multiplied to get the top of the mapping cylinder
 * @return Filtration of a mapping cylinder of the inclusion L \to K. Type of cells in the returned filtration is Cell \times Simplex.
 */
{
    using Int = typename Cell::Int;
    using ProdCell = ProductCell<Cell, Simplex<Int>>;
    using ResultCell = CellWithValue<ProdCell, Real>;
    using UidSet = typename Cell::UidSet;

    if (fil_domain.negate() != fil_codomain.negate()) {
        throw std::runtime_error("different negate values not supported");
    }

    if (v_domain.get_uid() == v_codomain.get_uid()) {
        throw std::runtime_error("cannot use same vertices for top and bottom of the cylinder");
    }

    if (v_domain.dim() != 0 or v_codomain.dim() != 0) {
        throw std::runtime_error("v_domain and v_codomain must be vertices (zero-dimensional simplices)");
    }

    auto f = get_inclusion_mapping<Cell, Real>(fil_domain, fil_codomain);

    bool is_surjective = fil_domain.size() == fil_codomain.size();

    Simplex<Int> edge {std::max(v_domain.get_id(), v_codomain.get_id()) + 1, {v_domain.vertices_[0], v_codomain.vertices_[0]}};

    std::vector<ResultCell> cyl_simplices;
    cyl_simplices.reserve(2 * fil_domain.size() + fil_codomain.size());

    // get top simplices
    append_products(fil_domain.cells(), v_domain, cyl_simplices);
    // get bottom simplices
    append_products(fil_codomain.cells(), v_codomain, cyl_simplices);
    // get cylinder interior simplices
    append_products(fil_domain.cells(), edge, cyl_simplices);

    if (not is_surjective) {
        // append codomain simplices not included in the domain
        UidSet image_uids;

        for(size_t sigma_idx = 0 ; sigma_idx < fil_domain.size() ; ++sigma_idx) {
            const auto& f_sigma = fil_codomain.get_cell(f[sigma_idx]);
            image_uids.insert(f_sigma.get_uid());
        }

        for(const auto& tau: fil_codomain.cells()) {
            if (image_uids.count(tau.get_uid()))
                continue;
            cyl_simplices.emplace_back(ProdCell(tau.get_cell(), v_codomain), tau.get_value());
        }
    }

    return Filtration<ProdCell, Real>(cyl_simplices, fil_domain.negate());
}

} // end of namespace oineus