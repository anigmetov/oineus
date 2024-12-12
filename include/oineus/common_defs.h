#pragma once

#include <cassert>
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <limits>

#include "log_wrapper.h"
#include "profile.h"

namespace oineus {

    using dim_type = size_t;
    using id_type = int;

    constexpr size_t plus_inf = std::numeric_limits<size_t>::max();

    constexpr size_t k_invalid_index = std::numeric_limits<size_t>::max();

template<typename Real>
    struct RPoint {
        Real x {0};
        Real y {0};

        RPoint() = default;

        RPoint(Real _x, Real _y)
                :x(_x), y(_y) { };

        const Real& operator[](int i) const
        {
            switch(i) {
            case 0:return x;
                break;
            case 1:return y;
                break;
            default:throw std::out_of_range("RPoint has only 2 coords");
            }
        }

        Real& operator[](int i)
        {
            switch(i) {
            case 0:return x;
                break;
            case 1:return y;
                break;
            default:throw std::out_of_range("RPoint has only 2 coords");
            }
        }
    };

    template<class T>
    inline void hash_combine(std::size_t& seed, const T& v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }


//template<typename Out>
//void split_by_delim(const std::string& s, char delim, Out result)
//{
//    std::stringstream ss(s);
//    std::string item;
//    while(std::getline(ss, item, delim)) {
//        *(result++) = item;
//    }
//}

//inline std::vector<std::string> split_by_delim(const std::string& s, char delim)
//{
//    std::vector<std::string> elems;
//    split_by_delim(s, delim, std::back_inserter(elems));
//    return elems;
//}
}
