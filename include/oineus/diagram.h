#pragma once

#include <vector>
#include <cmath>
#include <map>
#include <utility>
#include <string>
#include <ostream>
#include <fstream>
#include <stdexcept>

using namespace std::rel_ops;

namespace oineus {

template <typename T>
struct DgmPoint {
    T birth;
    T death;

    DgmPoint(T b, T d)
            :birth(b), death(d) { };

    T persistence() const { return std::abs(death - birth); }

    const T& operator[](int index) const
    {
        switch (index) {
        case 0 :
            return birth;
        case 1 :
            return death;
        default:
            throw std::out_of_range("DgmPoint has only 2 coordinates.");
        }
    }

    bool operator==(const DgmPoint& other) const { return birth == other.birth and death == other.death; }

    // compare by persistence first
    bool operator<(const DgmPoint& other) const
    {
        T pers = persistence();
        T other_pers = other.persistence();
        return std::tie(pers, birth, death) < std::tie(other_pers, other.birth, other.death);
    }

    // if we want indices, T will be integral and won't have infinity;
    // then lowest and max will be used to represent points at infinity
    static bool is_minus_inf(T x)
    {
        if (std::numeric_limits<T>::has_infinity)
            return x == -std::numeric_limits<T>::infinity();
        else
            return x == std::numeric_limits<T>::lowest();
    }

    static bool is_plus_inf(T x)
    {
        if (std::numeric_limits<T>::has_infinity)
            return x == std::numeric_limits<T>::infinity();
        else
            return x == std::numeric_limits<T>::max();
    }

    bool is_inf() const
    {
        return is_plus_inf(birth) or is_plus_inf(death) or is_minus_inf(birth) or is_minus_inf(death);
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const DgmPoint<T>& p)
{
    out << "(";
    if (DgmPoint<T>::is_minus_inf(p.birth))
        out << "-inf, ";
    else if (DgmPoint<T>::is_plus_inf(p.birth))
        out << "inf, ";
    else
        out << p.birth << ", ";

    if (DgmPoint<T>::is_minus_inf(p.death))
        out << "-inf";
    else if (DgmPoint<T>::is_plus_inf(p.death))
        out << "inf";
    else
        out << p.death;
    out << ")";

    return out;
}

template <typename Real_>
struct Diagrams {
    using Real = Real_;

    using Point = DgmPoint<Real>;
    using Dgm = std::vector<Point>;

    std::map<dim_type, Dgm> diagram_in_dimension_;

    // will throw, if there is no diagram for dimension d
    Dgm get_diagram_in_dimension(dim_type d) const
    {
        return diagram_in_dimension_.at(d);
    }

    void add_point(dim_type dim, Real b, Real d)
    {
        diagram_in_dimension_[dim].emplace_back(b, d);
    }

    void sort()
    {
        for (auto& dim_points : diagram_in_dimension_) {
            auto& points = dim_points.second;
            std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a > b; });
        }
    }

    void save_as_txt(std::string fname, std::string extension = "txt", int prec = 4) const
    {
        for (const auto& dim_points : diagram_in_dimension_) {

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

            for (const auto& p : points)
                f << p << "\n";

            f.close();
        }
    }
};

} // namespace oineus

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
