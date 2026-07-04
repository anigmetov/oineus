#pragma once

#include <cctype>
#include <cstdlib>
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

    static void set_level([[maybe_unused]] level::level_enum lvl) {}
    static void flush_on([[maybe_unused]] level::level_enum lvl) {}

    template<class... Args>
    static void trace([[maybe_unused]] Args... args) {}

    template<class... Args>
    static void debug([[maybe_unused]] Args... args) {}

    template<class... Args>
    static void info([[maybe_unused]] Args... args) {}

    template<class... Args>
    static void warn([[maybe_unused]] Args... args) {}

    template<class... Args>
    static void err([[maybe_unused]] Args... args) {}

    template<class... Args>
    static void critical([[maybe_unused]] Args... args) {}
};


template<class... T>
std::shared_ptr<logger> basic_logger_mt([[maybe_unused]] T...)
{
    return nullptr;
}


template<class... T>
std::shared_ptr<logger> stderr_color_mt([[maybe_unused]] T...)
{
    return nullptr;
}

template<class... T>
std::shared_ptr<logger> get([[maybe_unused]] T...)
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

namespace oineus {

// Log level of the internal reduction loggers, read once from the environment
// variable OINEUS_LOG_LEVEL (trace/debug/info/warn/error/critical/off; default
// info). Only meaningful in spdlog builds; spdlog never appears in the API.
inline spd::level::level_enum log_level_from_env()
{
    static const spd::level::level_enum level = [] {
        const char* env = std::getenv("OINEUS_LOG_LEVEL");
        std::string s = env ? env : "";
        for(auto& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s == "trace")    return spd::level::trace;
        if (s == "debug")    return spd::level::debug;
        if (s == "warn")     return spd::level::warn;
        if (s == "error")    return spd::level::err;
        if (s == "critical") return spd::level::critical;
        if (s == "off")      return spd::level::off;
        return spd::level::info;
    }();
    return level;
}

} // namespace oineus
