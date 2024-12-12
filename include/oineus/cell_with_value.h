#pragma once

#include <limits>
#include <vector>
#include <type_traits>

#include "common_defs.h"

namespace oineus {

template<class Cell_, class Real_>
struct CellWithValue {
    using Real = Real_;
    using Cell = Cell_;

    using Int = typename Cell::Int;
    using Uid = typename Cell::Uid;
    using UidSet = typename Cell::UidSet;
    using Boundary = typename Cell::Boundary;

    static constexpr Int k_invalid_id = Int(-1);

    Cell cell_ {};
    Int sorted_id_ {k_invalid_id};
    Real value_ {std::numeric_limits<Real>::max()};

    CellWithValue() = default;
    CellWithValue(const CellWithValue&) = default;
    CellWithValue(CellWithValue&&) = default;
    CellWithValue& operator=(const CellWithValue&) = default;
    CellWithValue& operator=(CellWithValue&&) = default;

//    template<class... Args>
//    CellWithValue(Real value, Args&&... args)
//            : cell_(std::forward(args)...), value_(value) { }

    CellWithValue(const Cell& cell, Real value) : cell_(cell), value_(value) {}
    CellWithValue(Cell&& cell, Real value) : cell_(std::move(cell)), value_(value) {}

    dim_type dim() const { return cell_.dim(); }

    Real get_value() const { return value_; }
    void set_value(Real new_value) { value_ = new_value; }

    Int get_id() const { return cell_.get_id(); }
    void set_id(Int new_id) { cell_.set_id(new_id); }

    Int get_sorted_id() const { return sorted_id_; }
    void set_sorted_id(Int sorted_id) { sorted_id_ = sorted_id; }

    Uid get_uid() const { return cell_.get_uid(); }
    void set_uid() { cell_.set_uid(); }

    const Cell& get_cell() const { return cell_; }

    Boundary boundary() const { return cell_.boundary(); }

    // create a new simplex by joining with vertex and assign value to it
    // will not compile, if Cell has no join method
    CellWithValue join(Int new_id, Int vertex, Real join_value) const
    {
        return CellWithValue(cell_.join(new_id, vertex), join_value);
    }

    auto get_factor_1() const
    {
        return cell_.get_factor_1();
    }

    auto get_factor_2() const
    {
        return cell_.get_factor_2();
    }

//    static CellWithValue create(Real value, const Cell& cell) { return CellWithValue(value, cell); }

    bool is_valid_filtration_simplex() const
    {
        return get_id() != k_invalid_id and get_sorted_id() != k_invalid_id;
    }

    bool operator==(const CellWithValue& other) const { return cell_ == other.cell_ and value_ == other.value_; }
    bool operator!=(const CellWithValue& other) const { return !(*this == other); }

    std::string repr() const
    {
        std::stringstream out;
        out << "CellWithValue(value_ =" << get_value() << ", sorted_id_ = " << get_sorted_id() << ", cell=" << cell_ << ")";
        return out.str();
    }

    template<typename R, typename C>
    friend std::ostream& operator<<(std::ostream&, const CellWithValue<R, C>&);
};

template<typename C, typename R>
std::ostream& operator<<(std::ostream& out, const CellWithValue<C, R>& s)
{
    out << "(" << s.get_cell() << ", " << s.get_value() << ")";
    return out;
}

} // namespace oineus