#pragma once

#include <string>

#ifdef OINEUS_USE_SPDLOG

#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/fmt/ostr.h>

#else

#include <memory>

// provide no-op wrappers for all spdlog functionality we need

namespace spdlog {
namespace level {
enum level_enum
{
    trace,
    debug,
    info,
    warn,
    err,
    critical,
    off
};
} // namespace level


struct logger {
    template<class... Args>
    logger(Args...) {}

    static void flush() {}

    static void set_level(level::level_enum lvl) {}
    static void flush_on(level::level_enum lvl) {}

    template<class... Args>
    static void trace(Args... args) {}

    template<class... Args>
    static void debug(Args... args) {}

    template<class... Args>
    static void info(Args... args) {}

    template<class... Args>
    static void warn(Args... args) {}

    template<class... Args>
    static void err(Args... args) {}

    template<class... Args>
    static void critical(Args... args) {}
};


template<class... T>
std::shared_ptr<logger> basic_logger_mt(T...)
{
    return nullptr;
}


template<class... T>
std::shared_ptr<logger> stderr_color_mt(T...)
{
    return nullptr;
}

template<class... T>
std::shared_ptr<logger> get(T...)
{
    return nullptr;
}

//std::shared_ptr<logger>
//get_logger()
//{
//    return nullptr;
//}
//
//level::level_enum
//get_log_level()
//{
//    return level::debug;
//}

//template<typename... Args>
//std::shared_ptr<spd::logger>
//set_logger(Args... args)
//{
//    auto log = std::make_shared<spdlog::logger>("oineus", args...);
//    return log;
//}

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



} // namespace spdlog

#endif

namespace spd=spdlog;
