#ifndef OINEUS_INTERRUPT_H
#define OINEUS_INTERRUPT_H

#include <csignal>
#include <exception>

namespace oineus {

// Inline so the header-only library has exactly one definition.
// volatile sig_atomic_t is the only type C/C++ guarantees may be
// safely read/written from an async signal handler.
inline volatile std::sig_atomic_t g_stop_flag = 0;

inline bool interrupted() noexcept { return g_stop_flag != 0; }

inline void request_stop() noexcept { g_stop_flag = 1; }
inline void clear_stop()   noexcept { g_stop_flag = 0; }

struct interrupted_exception : std::exception {
    const char* what() const noexcept override
    {
        return "oineus: computation interrupted";
    }
};

} // namespace oineus

#endif // OINEUS_INTERRUPT_H
