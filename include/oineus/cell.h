#pragma once

#include <limits>
#include <vector>
#include <set>
#include <ostream>
#include <algorithm>
#include <utility>


namespace oineus {

    template<typename Int_, typename Real_>
    struct Cell {
        // User-defined cell
        using Int = Int_;
        using Real = Real_;
        using IdxVector = std::vector<Int>;
        using Boundary = std::vector<IdxVector>;

        static constexpr Int k_invalid_id = Int(-1);

        Int id_ {k_invalid_id};
        Int sorted_id_ {k_invalid_id};
        IdxVector vertices_;
        Boundary boundary_;
        Real value_ {std::numeric_limits<Real>::max()};

        Cell() = default;

        Cell(const IdxVector& _vertices, const Boundary& _boundary, Real _value)
                :
                vertices_(_vertices), boundary_(_boundary), value_(_value)
        {
            if (vertices_.empty())
                throw std::runtime_error("Empty cell not allowed");

            if (vertices_.size() > 1)
                id_ = vertices_[0];
            else
                std::sort(vertices_.begin(), vertices_.end());
        }

        Int dim() const { return static_cast<Int>(vertices_.size()) - 1; }
        Real value() const { return value_; }

        Cell(const int _id, const IdxVector& _vertices, const Boundary& _boundary, Real _value)
                :
                id_(_id), vertices_(_vertices), boundary_(_boundary), value_(_value)
        {
            if (vertices_.empty())
                throw std::runtime_error("Empty cell not allowed");

            if (vertices_.size() > 1)
                std::sort(vertices_.begin(), vertices_.end());
        }


        Boundary boundary() const
        {
            return boundary_;
        }

        bool is_valid() const
        {
            std::set<Int> vertices_in_boundary;

            for(auto&& sigma : boundary_) {
                if (sigma.empty())
                    return false;

                if (not std::is_sorted(sigma.begin(), sigma.end())) {
                    return false;
                }

                if (sigma.size() + 1 != vertices_.size()) {
                    return false;
                }

                if (not std::includes(sigma.begin(), sigma.end(), vertices_.begin(), vertices_.end())) {
                    return false;
                }

                for(auto v : sigma) {
                    vertices_in_boundary.add(v);
                }
            }

            if (vertices_in_boundary.size() != vertices_.size()) {
                return false;
            }

            return true;
        }

        bool operator==(const Cell& other) const
        {
            // ignore id_ ?
            return sorted_id_ == other.sorted_id_ and vertices_ == other.vertices_ and value_ == other.value_;
        }

        template<typename I, typename R, typename L>
        friend std::ostream& operator<<(std::ostream&, const Cell<I, R>&);

        Int get_sorted_id()
        {
            return sorted_id_;
        }
    };

    template<typename I, typename R>
    std::ostream& operator<<(std::ostream& out, const Cell<I, R>& s)
    {
        out << "Cell(id_=" << s.id_ << ", sorted_id_ = " << s.sorted_id_ << ", vertices_=(";

        for(size_t i = 0; i < s.vertices_.size() - 1; ++i)
            out << s.vertices_[i] << ", ";

        out << s.vertices_[s.vertices_.size() - 1] << "), value_=" << s.value_ << ")";

        return out;
    }
} // namespace oineus
