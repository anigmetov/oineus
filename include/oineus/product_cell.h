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
build_mapping_cylinder(const Filtration<Cell, Real>& fil_domain, const Filtration<Cell, Real>& fil_target, typename Cell::Int id_v_domain, typename Cell::Int id_v_codomain)
{
    using Int = typename Cell::Int;
    using CWV = CellWithValue<Cell, Real>;
    using ProdCell = ProductCell<Cell, Simplex<Int>>;
    using ResultCell = CellWithValue<ProdCell, Real>;
    using UidSet = typename Cell::UidSet;

    if (fil_domain.negate() != fil_target.negate()) {
        throw std::runtime_error("different negate values not supported");
    }

    if (id_v_domain == id_v_codomain) {
        throw std::runtime_error("cannot use same vertices for top and bottom of the cylinder");
    }

    auto f = get_inclusion_mapping<Cell, Real>(fil_domain, fil_target);

    bool is_surjective = fil_domain.size() == fil_target.size();

    Simplex<Int> v0 {id_v_domain, {id_v_domain}};
    Simplex<Int> v1 {id_v_domain, {id_v_codomain}};
    Simplex<Int> e01 {std::max(id_v_domain, id_v_codomain) + 1, {id_v_domain, id_v_codomain}};

    std::vector<ResultCell> cyl_simplices;
    cyl_simplices.reserve(2 * fil_domain.size() + fil_target.size());

    // get top simplices
    append_products(fil_domain.cells(), v0, cyl_simplices);
    // get bottom simplices
    append_products(fil_target.cells(), v1, cyl_simplices);
    // get cylinder interior simplices
    append_products(fil_domain.cells(), e01, cyl_simplices);

    if (not is_surjective) {
        // append codomain simplices not included in the domain
        UidSet image_uids;

        for(size_t sigma_idx = 0 ; sigma_idx < fil_domain.size() ; ++sigma_idx) {
            const auto& f_sigma = fil_target.get_cell(f[sigma_idx]);
            image_uids.insert(f_sigma.get_uid());
        }

        for(const auto& tau: fil_target.cells()) {
            if (image_uids.count(tau.get_uid()))
                continue;
            cyl_simplices.emplace_back(ProdCell(tau.get_cell(), v1), tau.get_value());
        }
    }

    return Filtration<ProdCell, Real>(cyl_simplices, fil_domain.negate());
}

} // end of namespace oineus