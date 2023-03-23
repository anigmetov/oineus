#pragma once

#include <vector>
#include <cmath>
#include <map>
#include <utility>
#include <string>
#include <ostream>
#include <fstream>
#include <stdexcept>

namespace oineus {

    template<typename T>
    struct DgmPoint {
        T birth;
        T death;

        id_type id {0};

        DgmPoint() = default;
        DgmPoint(const DgmPoint&) = default;

        DgmPoint(T b, T d)
                :birth(b), death(d) { };

        DgmPoint(T b, T d, id_type i)
                :birth(b), death(d), id(i) { };

        T persistence() const { return std::abs(death - birth); }

        T& operator[](int index)
        {
            switch(index) {
            case 0 :return birth;
            case 1 :return death;
            default:throw std::out_of_range("DgmPoint has only 2 coordinates.");
            }
        }

        const T& operator[](int index) const
        {
            switch(index) {
            case 0 :return birth;
            case 1 :return death;
            default:throw std::out_of_range("DgmPoint has only 2 coordinates.");
            }
        }

        // if we want indices, T will be integral and won't have infinity;
        // then lowest and max will be used to represent points at infinity
        static bool is_minus_inf(T x) { return x == minus_inf(); }
        static bool is_plus_inf(T x) { return x == plus_inf(); }

        static constexpr T plus_inf()
        {
            if constexpr (std::numeric_limits<T>::has_infinity)
                return std::numeric_limits<T>::infinity();
            else
                return std::numeric_limits<T>::max();
        }

        static constexpr T minus_inf()
        {
            if constexpr (std::numeric_limits<T>::has_infinity)
                return -std::numeric_limits<T>::infinity();
            else
                return std::numeric_limits<T>::lowest();
        }

        bool is_inf() const
        {
            return is_plus_inf(birth) or is_plus_inf(death) or is_minus_inf(birth) or is_minus_inf(death);
        }
    };

    template<class R>
    bool operator==(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        return a.birth == b.birth and a.death == b.death;
    }

    template<class R>
    bool operator!=(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        return !(a == b);
    }

// compare by persistence first
    template<class R>
    bool operator<(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        R pers = a.persistence();
        R other_pers = b.persistence();
        return std::tie(pers, a.birth, a.death) < std::tie(other_pers, b.birth, b.death);
    }

    template<class R>
    bool operator<=(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        return a < b or a == b;
    }

    template<class R>
    bool operator>(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        R pers = a.persistence();
        R other_pers = b.persistence();
        return std::tie(pers, a.birth, a.death) > std::tie(other_pers, b.birth, b.death);
    }

    template<class R>
    bool operator>=(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        return a > b or a == b;
    }

    template<typename T>
    std::string to_string_possible_inf(const T& a)
    {
        if (DgmPoint<T>::is_minus_inf(a))
            return "-inf, ";
        else if (DgmPoint<T>::is_plus_inf(a))
            return "inf, ";
        else
            return std::to_string(a);
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& out, const DgmPoint<T>& p)
    {
        out << "(" << to_string_possible_inf<T>(p.birth) << ", " << to_string_possible_inf<T>(p.death) << ", id = " << p.id << ")";
        return out;
    }

    template<typename Real_>
    struct Diagrams {
        using Real = Real_;

        using Point = DgmPoint<Real>;
        using Dgm = std::vector<Point>;

        dim_type max_dim_;

        Diagrams(dim_type filtration_dim)
                :max_dim_(filtration_dim - 1)
        {
            if (filtration_dim == 0)
                throw std::runtime_error("refuse to compute diagram from 0-dim filtration");

            for(dim_type d = 0; d <= max_dim_; ++d)
                diagram_in_dimension_[d];
        }

        std::map<dim_type, Dgm> diagram_in_dimension_;

        [[nodiscard]] size_t n_dims() const noexcept { return diagram_in_dimension_.size(); }

        // will throw, if there is no diagram for dimension d
        Dgm get_diagram_in_dimension(dim_type d) const
        {
            return diagram_in_dimension_.at(d);
        }

        Dgm& operator[](size_t d) { return diagram_in_dimension_.at(d); }
        const Dgm& operator[](size_t d) const { return diagram_in_dimension_.at(d); }

        void add_point(dim_type dim, Real b, Real d)
        {
            diagram_in_dimension_[dim].emplace_back(b, d);
        }

        void sort()
        {
            for(auto& dim_points: diagram_in_dimension_) {
                auto& points = dim_points.second;
                std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a > b; });
            }
        }

        void save_as_txt(std::string fname, std::string extension = "txt", int prec = 4) const
        {
            for(const auto& dim_points: diagram_in_dimension_) {

                const auto& points = dim_points.second;

                if (points.empty())
                    continue;

                auto dim = dim_points.first;

                std::string fname_dim = fname + "." + std::to_string(dim) + extension;

                std::ofstream f(fname_dim);
                if (!f.good()) {
                    std::cerr << "Cannot open file " << fname_dim << std::endl;
                    throw std::runtime_error("Cannot write diagram");
                }

                f.precision(prec);

                for(const auto& p: points)
                    f << p << "\n";

                f.close();
            }
        }
    };

} // namespace oineus


namespace std {
    template<class T>
    struct hash<oineus::DgmPoint<T>> {
        std::size_t operator()(const oineus::DgmPoint<T>& p) const
        {
            std::size_t seed = 0;
            oineus::hash_combine(seed, p.birth);
            oineus::hash_combine(seed, p.death);
            oineus::hash_combine(seed, p.id);
            return seed;
        }
    };
};


//  template<class Cont>
//    std::string container_to_string(const Cont& v)
//    {
//        std::stringstream ss;
//        ss << "[";
//        for(auto x_iter = v.begin(); x_iter != v.end(); ) {
//            ss << *x_iter;
//            x_iter = std::next(x_iter);
//            if (x_iter != v.end())
//                ss << ", ";
//        }
////        for(const auto& x : v) {
////            ss << x << ", ";
////        }
//        ss << "]";
//        return ss.str();
//    }
