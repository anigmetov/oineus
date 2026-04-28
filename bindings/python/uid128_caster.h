#pragma once

// nanobind type caster for unsigned __int128, used as the storage type for
// Simplex<Int>::Uid (see include/oineus/simplex.h). Converts to/from a Python
// arbitrary-precision int as (hi << 64) | lo. Once this header is included,
// every binding signature involving unsigned __int128 -- function arguments,
// return values, and STL container element types (e.g. std::unordered_map
// keyed by Uid) -- is auto-converted on the Python boundary without any
// per-binding lambda.

#include <cstdint>

#include <nanobind/nanobind.h>

#if !defined(__SIZEOF_INT128__)
#  error "uid128_caster.h requires unsigned __int128 (gcc / clang)."
#endif

namespace nanobind::detail {

template <>
struct type_caster<unsigned __int128> {
    NB_TYPE_CASTER(unsigned __int128, const_name("int"))

    bool from_python(handle src, uint8_t /*flags*/, cleanup_list* /*cl*/) noexcept
    {
        if (!PyLong_Check(src.ptr())) return false;
        try {
            int_ x = borrow<int_>(src);
            int_ mask((std::uint64_t) ~0ULL);
            std::uint64_t lo = nanobind::cast<std::uint64_t>(x & mask);
            std::uint64_t hi = nanobind::cast<std::uint64_t>(x >> int_(64));
            value = (static_cast<unsigned __int128>(hi) << 64) | lo;
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle from_cpp(unsigned __int128 v, rv_policy /*policy*/, cleanup_list* /*cl*/) noexcept
    {
        try {
            int_ lo(static_cast<std::uint64_t>(v));
            int_ hi(static_cast<std::uint64_t>(v >> 64));
            return ((hi << int_(64)) | lo).release();
        } catch (...) {
            return handle();
        }
    }
};

}  // namespace nanobind::detail
