#pragma once

#include <cassert>
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include <sstream>

#ifdef ARACHNE_USE_SPDLOG

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
namespace spd = spdlog;

template<typename... Args>
inline void debug(const Args& ... args)
{
    spd::debug(args...);
}

template<typename... Args>
inline void info(const Args& ... args)
{
    spd::info(args...);
}

#else

template<typename... Args>
inline void debug([[maybe_unused]] const Args& ... args)
{
    ;
}

template<typename... Args>
inline void info([[maybe_unused]] const Args& ... args)
{
    ;
}

#endif

namespace oineus {

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

    struct Params {
        int n_threads {1};
        int chunk_size {128};
        bool write_dgms {false};
        bool sort_dgms {true};
        bool clearing_opt {false};
        bool acq_rel {false};
        bool print_time {true};
        double elapsed { 0.0 };
    };

    struct ThreadStats {
        const int thread_id;
        long int n_right_pivots {0};
        long int n_cleared {0};

        ThreadStats() : thread_id(-1) { }

        ThreadStats(int _thread_id) : thread_id(_thread_id) {}
    };
}
