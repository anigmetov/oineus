#pragma once

// Author: Dmitriy Morozov

#include <chrono>

struct Timer
{
    //using clock = std::chrono::steady_clock;
    using clock = std::chrono::high_resolution_clock;
    using second = std::chrono::duration<double, std::ratio<1>>;
    using time = std::chrono::time_point<clock>;

    time last_;

    Timer():
        last_(clock::now())     {}

    void    reset()             { last_ = clock::now(); }
    double  elapsed()           { return std::chrono::duration_cast<second> (clock::now() - last_).count(); }
    double  elapsed_reset()     { time last = last_; last_ = clock::now(); auto diff = std::chrono::duration_cast<second> (last_ - last).count(); return diff; }
};
