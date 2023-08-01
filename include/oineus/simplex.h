#pragma once

#include <limits>
#include <vector>
#include <ostream>
#include <algorithm>
#include <utility>

//using namespace std::rel_ops;

namespace oineus {

    struct VREdge {
        size_t x;
        size_t y;
        bool operator==(const VREdge& other) const { return x == other.x and y == other.y; }
        bool operator!=(const VREdge& other) const { return !(*this == other); }
        bool operator<(const VREdge& other) const { return x < other.x or (x == other.x and y < other.y); };
        bool operator>(const VREdge& other) const { return other < *this; }
        bool operator<=(const VREdge& other) const { return *this < other or *this == other; }
        bool operator>=(const VREdge& other) const { return *this > other or *this == other; }
        // rel_ops takes care of other comparison operators are
    };

    std::ostream& operator<<(std::ostream& out, const VREdge& e)
    {
        out << "edge(x=" << e.x << ", y=" << e.y << ")";
        return out;
    }

    template<typename Int_, typename Real_>
    struct Simplex {
        using Int = Int_;
        using Real = Real_;
        using IdxVector = std::vector<Int>;

        Int user_data;

        IdxVector vertices_;
        Real value_ {std::numeric_limits<Real>::max()};

        Simplex() = default;

        Simplex(const IdxVector& _vertices, Real _value)
                :
                vertices_(_vertices), value_(_value)
        {
            if (vertices_.empty())
                throw std::runtime_error("Empty simplex not allowed");

            if (vertices_.size() > 1)
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

        // create a new simplex by joining with vertex and assign value to it
        Simplex join(Int vertex, Real value) const
        {
            // new vertex must not be present in this->vertices
            if (std::find(vertices_.begin(), vertices_.end(),vertex) != vertices_.end())
                throw std::runtime_error("Duplicate vertex in join");

            IdxVector vs = vertices_;
            vs.push_back(vertex);
            return Simplex(vs, value);
        }


        bool operator==(const Simplex& other) const
        {
            return vertices_ == other.vertices_ and value_ == other.value_;
        }

        template<typename I, typename R, typename L>
        friend std::ostream& operator<<(std::ostream&, const Simplex<I, R>&);
    };

    template<typename I, typename R>
    std::ostream& operator<<(std::ostream& out, const Simplex<I, R>& s)
    {
        out << "Simplex(vertices_=(";

        for(size_t i = 0; i < s.vertices_.size() - 1; ++i)
            out << s.vertices_[i] << ", ";

        out << s.vertices_[s.vertices_.size() - 1] << "), value_=" << s.value_ << ", user_data = " << s.user_data << ")";

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
