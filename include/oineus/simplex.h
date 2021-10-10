#pragma once

#include <limits>
#include <vector>
#include <ostream>
#include <algorithm>
#include <utility>

using namespace std::rel_ops;

namespace oineus {

struct VREdge {
    size_t x;
    size_t y;
    bool operator==(const VREdge& other) const { return x == other.x and y == other.y; }
    bool operator<(const VREdge& other) const { return x < other.x or (x == other.x and y < other.y); };
    // rel_ops takes care of other comparison operators are
};

std::ostream& operator<<(std::ostream& out, const VREdge& e)
{
    out << "edge(x=" << e.x << ", y=" << e.y << ")";
    return out;
}

template<typename Int_, typename Real_, typename ValueLocation>
struct Simplex {
    using Int = Int_;
    using Real = Real_;
    using IdxVector = std::vector<Int>;

    static constexpr Int k_invalid_id = Int(-1);

    Int id_ {k_invalid_id};
    Int sorted_id_ {k_invalid_id};
    IdxVector vertices_;
    Real value_ {std::numeric_limits<Real>::max()};
    ValueLocation critical_value_location_;

    Simplex() = default;

    Simplex(const IdxVector& _vertices, Real _value, const ValueLocation& _critical_value_location)
            :
            vertices_(_vertices), value_(_value), critical_value_location_(_critical_value_location)
    {
        if (vertices_.empty())
            throw std::runtime_error("Empty simplex not allowed");

        if (vertices_.size() == 1)
            id_ = vertices_[0];
        else
            std::sort(vertices_.begin(), vertices_.end());
    }

    Int dim() const { return static_cast<Int>(vertices_.size()) - 1; }
    Real value() const { return value_; }

    std::vector<IdxVector> boundary() const
    {
        std::vector<IdxVector> bdry;
        bdry.reserve(vertices_.size());

        for(size_t i = 0; i < vertices_.size(); ++i) {
            IdxVector tau;
            tau.reserve(vertices_.size() - 1);

            for(size_t j = 0; j < vertices_.size(); ++j)
                if (j != i)
                    tau.push_back(vertices_[j]);

            // vertices_ is sorted -> tau is sorted automatically

            bdry.push_back(tau);
        }

        return bdry;
    }

    bool is_valid_filtration_simplex() const
    {
        return id_ != k_invalid_id and sorted_id_ != k_invalid_id;
    }

    bool operator==(const Simplex& other) const
    {
        // ignore id_ ?
        return sorted_id_ == other.sorted_id_ and vertices_ == other.vertices_ and value_ == other.value_ and critical_value_location_ == other.critical_value_location_;
    }

    template<typename I, typename R, typename L>
    friend std::ostream& operator<<(std::ostream&, const Simplex<I, R, L>&);
};

template<typename I, typename R, typename L>
std::ostream& operator<<(std::ostream& out, const Simplex<I, R, L>& s)
{
    out << "Simplex(id_=" << s.id_ << ", sorted_id_ = " << s.sorted_id_ << ", vertices_=(";

    for(size_t i = 0; i < s.vertices_.size() - 1; ++i)
        out << s.vertices_[i] << ", ";

    out << s.vertices_[s.vertices_.size() - 1] << "), value_=" << s.value_ << ", value_location=" << s.critical_value_location_ << ")";

    return out;
}
}

namespace std {
template<>
struct hash<oineus::VREdge> {
    std::size_t operator()(const oineus::VREdge& p) const
    {
        std::size_t seed = 0;
        oineus::hash_combine(seed, p.x);
        oineus::hash_combine(seed, p.y);
        return seed;
    }
};
};