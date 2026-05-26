#ifndef OINEUS_SIGNAL_GUARD_H
#define OINEUS_SIGNAL_GUARD_H

// RAII Ctrl-C support for Oineus Python bindings.
//
// Plugged into nanobind's call_guard so every long computation responds
// to KeyboardInterrupt within tens of milliseconds:
//
//   .def("foo", &Class::foo,
//        nb::call_guard<nb::gil_scoped_release,
//                       oineus_python::SignalGuard>())
//
// Order is load-bearing. nanobind's call_guard is implemented as
// detail::tuple<Ts...> where tuple<T, Ts...> inherits from tuple<Ts...>,
// so the LAST template argument is constructed FIRST (it's at the base of
// the inheritance chain) and destructed LAST. With SignalGuard as the
// last argument, its destructor therefore runs AFTER gil_scoped_release
// has reacquired the GIL -- which is required because the destructor may
// call PyErr_SetString. The earlier reversed order caused intermittent
// segfaults: SignalGuard's destructor was running with the GIL still
// released, and PyErr_SetString without the GIL is undefined behavior.
//
// The destructor must be noexcept -- if a C++ exception is already
// propagating (the common case when interrupt is detected), throwing
// from a destructor calls std::terminate.
//
// The actual translation to KeyboardInterrupt happens via a nanobind
// exception translator registered at module init time (see
// register_interrupt_translator below): C++ code throws
// oineus::interrupted_exception when it sees the global stop flag, and
// the translator catches it and sets PyExc_KeyboardInterrupt.
//
// SignalGuard's job is therefore just (a) install/restore the C-level
// SIGINT handler (ref-counted across concurrent calls) and (b) handle
// the corner case where the flag was set but the workload finished
// without ever throwing -- in that case we set the indicator directly
// so the caller still sees KeyboardInterrupt.

#include <atomic>
#include <csignal>
#include <exception>
#include <mutex>

#include <nanobind/nanobind.h>
#include <nanobind/nb_python.h>

#include <oineus/interrupt.h>

namespace oineus_python {

namespace nb = nanobind;

namespace detail {

inline std::mutex& install_mutex()
{
    static std::mutex m;
    return m;
}

inline int& install_refcount()
{
    static int n = 0;
    return n;
}

inline struct sigaction& saved_sigint()
{
    static struct sigaction sa{};
    return sa;
}

extern "C" inline void oineus_sigint_handler(int)
{
    oineus::g_stop_flag = 1;
}

} // namespace detail

class SignalGuard
{
public:
    SignalGuard() noexcept
    {
        std::lock_guard<std::mutex> lock(detail::install_mutex());
        if (detail::install_refcount()++ == 0) {
            oineus::clear_stop();
            struct sigaction sa{};
            sa.sa_handler = &detail::oineus_sigint_handler;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0; // do not set SA_RESTART; let blocking syscalls fail
            sigaction(SIGINT, &sa, &detail::saved_sigint());
        }
    }

    // Not noexcept: on the success path we may need to raise
    // KeyboardInterrupt by throwing nb::python_error{}, so nanobind sees
    // a thrown exception (otherwise it sees a returned value with the
    // error indicator set and reports SystemError). During stack
    // unwinding (uncaught_exceptions() > 0) we skip the throw so we
    // don't replace the in-flight C++ exception (terminate would
    // result).
    ~SignalGuard() noexcept(false)
    {
        bool need_raise = false;
        {
            std::lock_guard<std::mutex> lock(detail::install_mutex());
            if (--detail::install_refcount() == 0) {
                sigaction(SIGINT, &detail::saved_sigint(), nullptr);
                if (std::uncaught_exceptions() == 0 && oineus::interrupted()) {
                    oineus::clear_stop();
                    need_raise = true;
                }
            }
        }
        if (need_raise) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_KeyboardInterrupt, "");
            throw nb::python_error{};
        }
    }

    SignalGuard(const SignalGuard&) = delete;
    SignalGuard& operator=(const SignalGuard&) = delete;
};

inline void register_interrupt_translator()
{
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* /*payload*/) {
            try {
                std::rethrow_exception(p);
            } catch (const oineus::interrupted_exception&) {
                PyErr_SetString(PyExc_KeyboardInterrupt, "");
            }
        },
        nullptr);
}

} // namespace oineus_python

#endif // OINEUS_SIGNAL_GUARD_H
