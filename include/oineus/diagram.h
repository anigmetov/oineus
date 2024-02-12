#pragma once

#include <vector>
#include <cmath>
#include <map>
#include <utility>
#include <string>
#include <ostream>
#include <fstream>
#include <stdexcept>

#include "common_defs.h"

namespace oineus {

    template<typename T>
    struct DgmPoint {
        T birth;
        T death;

        size_t birth_index {k_invalid_index};
        size_t death_index {k_invalid_index};
		size_t birth_index_unsorted {k_invalid_index};
        size_t death_index_unsorted {k_invalid_index};

        id_type id {0};

        DgmPoint() = default;
        DgmPoint(const DgmPoint&) = default;
        DgmPoint(DgmPoint&&) noexcept = default;
        DgmPoint& operator=(const DgmPoint&) = default;
        DgmPoint& operator=(DgmPoint&&) noexcept = default;

        DgmPoint(T b, T d)
                :birth(b), death(d) { };

        DgmPoint(T b, T d, size_t b_i, size_t d_i)
                :birth(b), death(d), birth_index(b_i), death_index(d_i) { };

		DgmPoint(T b, T d, size_t b_i, size_t d_i, size_t b_i_us, size_t d_i_us)
                :birth(b), death(d), birth_index(b_i), death_index(d_i), birth_index_unsorted(b_i_us), death_index_unsorted(d_i_us) { };

        T persistence() const { return std::abs(death - birth); }
        size_t index_persistence() const { return std::abs(static_cast<long>(death_index) - static_cast<long>(birth_index)); }

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

        bool is_diagonal() const
        {
            return birth == death;
        }

    };

    template<class R>
    bool operator==(const DgmPoint<R>& a, const DgmPoint<R>& b)
    {
        return a.birth == b.birth and a.death == b.death and a.birth_index == b.birth_index and a.death_index == b.death_index;
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
        return std::tie(pers, a.birth, a.death, a.birth_index, a.death_index) < std::tie(other_pers, b.birth, b.death, b.birth_index, b.death_index);
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
        return std::tie(pers, a.birth, a.death, a.birth_index, a.death_index) > std::tie(other_pers, b.birth, b.death, b.birth_index, b.death_index);
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
            return "-inf";
        else if (DgmPoint<T>::is_plus_inf(a))
            return "inf";
        else
            return std::to_string(a);
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& out, const DgmPoint<T>& p)
    {
        out << "(" << to_string_possible_inf<T>(p.birth) << ", " << to_string_possible_inf<T>(p.death) << ", birth_index=" << p.birth_index << ", death_index=" << p.death_index << ")";
        return out;
    }

    template<typename Real_>
    struct Diagrams {
        using Real = Real_;

        using Point = DgmPoint<Real>;
        using Dgm = std::vector<Point>;
        using IndexDgm = std::vector<DgmPoint<size_t>>;

        dim_type max_dim_;

        Diagrams() { };

        Diagrams(dim_type filtration_dim)
                :max_dim_(filtration_dim - 1)
        {
            if (filtration_dim == 0)
                throw std::runtime_error("refuse to compute diagram from 0-dim filtration");

            for(dim_type d = 0; d <= max_dim_; ++d) {
                diagram_in_dimension_[d];
            }
        }

        Diagrams(const Diagrams&) = default;
        Diagrams(Diagrams&&) noexcept = default;
        Diagrams& operator=(const Diagrams&) = default;
        Diagrams& operator=(Diagrams&&) noexcept = default;

        std::map<dim_type, Dgm> diagram_in_dimension_;

        [[nodiscard]] size_t n_dims() const noexcept { return diagram_in_dimension_.size(); }

        Dgm get_diagram_in_dimension(dim_type d)
        {
            return diagram_in_dimension_[d];
        }

        IndexDgm get_index_diagram_in_dimension(dim_type d, bool sorted = true) const
        {
            IndexDgm index_dgm;

            // just duplicate information for now: birth and death are also indices
            if (sorted) {
				for(Point p : diagram_in_dimension_.at(d))
                	index_dgm.emplace_back(p.birth_index, p.death_index, p.birth_index, p.death_index);
			} else {
				std::cout << "sorted is set to false, so will return the original id" << std::endl;
				for(Point p : diagram_in_dimension_.at(d))
                	index_dgm.emplace_back(p.birth_index_unsorted, p.death_index_unsorted);
			}
            return index_dgm;
        }

        Dgm& extract(int i) { return diagram_in_dimension_[i]; }
        Dgm& operator[](size_t d) { return diagram_in_dimension_[d]; }
        const Dgm& operator[](size_t d) const { return diagram_in_dimension_.at(d); }

		void add_point(dim_type dim, Real birth_value, Real death_value, size_t birth_index, size_t death_index)
        {
            diagram_in_dimension_[dim].emplace_back(birth_value, death_value, birth_index, death_index);
        }

        void add_point(dim_type dim, Real birth_value, Real death_value, size_t birth_index, size_t death_index, size_t birth_index_unsorted, size_t death_index_unsorted)
        {
            diagram_in_dimension_[dim].emplace_back(birth_value, death_value, birth_index, death_index, birth_index_unsorted, death_index_unsorted);
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
            oineus::hash_combine(seed, p.birth_index);
            oineus::hash_combine(seed, p.death_index);
//            oineus::hash_combine(seed, p.id);
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
