#pragma once

#include <limits>
#include <vector>
#include <ostream>
#include <algorithm>

namespace oineus {

    template<typename Int_, typename Real_>
    struct Simplex {
        using Int = Int_;
        using Real = Real_;
        using IdxVector = std::vector<Int>;

        static constexpr Int k_invalid_id = Int(-1);

        Int id_ {k_invalid_id};
        Int sorted_id_ {k_invalid_id};
        IdxVector vertices_;
        Real value_ {std::numeric_limits<Real>::max()};

        Simplex() = default;

        Simplex(const IdxVector& _vertices, Real _value = std::numeric_limits<Real>::max()) :
            vertices_(_vertices), value_(_value)
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

        template<typename I, typename R>
        friend std::ostream& operator<<(std::ostream&, const Simplex<I, R>&);
    };

    template<typename I, typename R>
    std::ostream& operator<<(std::ostream& out, const Simplex<I, R>& s)
    {
        out << "Simplex(id_=" << s.id_ << ", sorted_id_ = " << s.sorted_id_ << ", vertices_=(";

        for(size_t i = 0; i < s.vertices_.size() - 1; ++i)
            out << s.vertices_[i] << ", ";

        out << s.vertices_[s.vertices_.size() - 1] << "), value_=" << s.value_ << ")";

        return out;
    }
}


