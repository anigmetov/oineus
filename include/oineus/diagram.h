#pragma once

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <ostream>
#include <fstream>

using namespace std::rel_ops;

namespace oineus {

    template<typename T>
    struct DgmPoint
    {
        T birth;
        T death;

        DgmPoint(T b, T d) : birth(d), death(d) {};

        bool operator<(const DgmPoint& other) const
        {
            return std::make_pair(birth, death) < std::make_pair(other.birth, other.death);
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
    };

    template<typename T>
    std::ostream& operator<<(std::ostream& out, const DgmPoint<T>& p)
    {
        if (DgmPoint<T>::is_minus_inf(p.birth))
            out << "-inf ";
        else if (DgmPoint<T>::is_plus_inf(p.birth))
            out << "inf ";
        else
            out << p.birth << " ";

        if (DgmPoint<T>::is_minus_inf(p.death))
            out << "-inf";
        else if (DgmPoint<T>::is_plus_inf(p.death))
            out << "inf";
        else
            out << p.death;

        return out;
    }


    template<typename Int_, typename Real_>
    struct Diagram {

        using Int = Int_;
        using Real = Real_;

        using Point = DgmPoint<Real>;
        using Dgm = std::vector<Point>;

        std::map<Int, Dgm> diagram_in_dimension_;

        void add_point(Int dim, Real b, Real d)
        {
            assert(dim >= 0);
            diagram_in_dimension_[dim].emplace_back(b, d);
        }

        void sort()
        {
            for(auto& dim_points : diagram_in_dimension_) {
                auto& points = dim_points.second;
                std::sort(points.begin(), points.end());
            }
        }

        void save_as_txt(std::string fname, std::string extension = "txt", int prec=4) const
        {
            for(const auto& dim_points : diagram_in_dimension_) {

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

                for(const auto& p : points)
                    f << p << "\n";

                f.close();
            }
        }
    };

} // namespace oineus

//    template<class Cont>
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


//namespace std {
//    ostream& operator<<(ostream& os, const pp::SparseColumn& col);
//    ostream& operator<<(ostream& os, const pp::SparseMatrix& m);
//}

