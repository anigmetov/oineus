//
// Created by narn on 5/4/24.
//

#ifndef OINEUS_MIN_FILTRATION_H
#define OINEUS_MIN_FILTRATION_H

#include <utility>
#include <cassert>

#include "filtration.h"

namespace oineus {

template<class C, class R>
std::tuple<Filtration<C, R>, std::vector<size_t>, std::vector<size_t>> min_filtration(const Filtration<C, R>& fil_1, const Filtration<C, R>& fil_2)
{
    using Fil = Filtration<C, R>;
    using Cell = typename Fil::Cell;
    using Real = R;

    assert(fil_1.size() == fil_2.size());
    assert(fil_1.negate() == fil_2.negate());

    bool negate = fil_1.negate();

    std::vector<Real> values_1 = fil_1.all_values();
    std::vector<Real> values_2 = fil_2.all_values();

    const std::vector<Cell>& cells = fil_1.cells();

    std::vector<std::tuple<Real, dim_type, size_t, size_t>> to_sort;

    to_sort.reserve(fil_1.size());

    for(size_t idx_1 = 0; idx_1 < fil_1.size(); ++idx_1) {
        size_t idx_2 = fil_2.get_sorted_id_by_uid(cells[idx_1].get_uid());

        Real value_1 = values_1[idx_1];
        Real value_2 = values_2[idx_2];

        Real min_value = negate ? std::min(-value_1, -value_2) : std::min(value_1, value_2);
        to_sort.emplace_back(min_value, cells[idx_1].dim(), idx_1, idx_2);
    }

    // lexicographic comparison: first by value, then by dimension
    std::sort(to_sort.begin(), to_sort.end());

    std::vector<Cell> min_cells;

    min_cells.reserve(fil_1.size());

    std::vector<size_t> perm_1, perm_2;

    for(const auto& [value, dim, index_1, index_2] : to_sort) {
        if (negate) {
            value = -value;
        }

        min_cells.emplace_back(cells[index_1]);
        min_cells.set_value(value);

        perm_1.push_back(index_1);
        perm_2.push_back(index_2);
    }

    Filtration<Cell, Real> new_fil(cells, negate);

    return { new_fil, perm_1, perm_2 };
}


}

#endif //OINEUS_MIN_FILTRATION_H
