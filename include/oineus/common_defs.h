#pragma once

#include <cassert>
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <new>
#include <type_traits>

#include "log_wrapper.h"
#include "profile.h"

#ifdef OINEUS_USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#endif

namespace oineus {

    using dim_type = size_t;
    using DimVec = std::vector<dim_type>;
    using id_type = int;

    // Tag type used by ctors that accept already-sorted / already-ordered
    // input and want to skip the redundant ordering step. Used by both
    // Simplex (skip vertex sort) and Filtration (skip the global sort).
    struct presorted_t {};
    inline constexpr presorted_t presorted{};

    // Empty geometry for cells that are self-contained -- they carry all the
    // information their (co)boundary needs (e.g. Simplex stores its vertices).
    // Cube, by contrast, needs the shared GridDomain, which the Filtration owns.
    // Every cell exposes `using Geometry = ...`; the Filtration stores one
    // Geometry and threads it into the (co)boundary policy. NoGeometry makes that
    // uniform: cells that ignore geometry declare `using Geometry = NoGeometry`
    // and accept an unused argument.
    struct NoGeometry {
        bool operator==(const NoGeometry&) const { return true; }
        bool operator!=(const NoGeometry&) const { return false; }
    };

    // Cell-policy traits the Filtration `if constexpr`-dispatches on. All default
    // false (e.g. the fat Simplex); a packed cell specializes the ones it supports.
    // Keyed on the underlying cell type, not CellWithValue. They are three ORTHOGONAL
    // concerns -- a bit-packed VR simplex, for example, will want the first (fast
    // buffer boundary) but not the others (its uid is sparse -> hash index; it has no
    // cheap direct coboundary -> antitranspose). Cube and the Freudenthal cell
    // specialize all three true.
    //
    // HasPackedBoundary: provides boundary_into(geometry, emit) -- emit(face_uid) per
    //   facet, no intermediate std::vector (the Stage 1b alloc-elision win).
    template<class Cell>
    struct HasPackedBoundary : std::false_type {};
    // HasDirectCoboundary: provides coboundary_into(geometry, emit), so the filtration
    //   builds the cohomology matrix directly instead of via a global antitranspose.
    template<class Cell>
    struct HasDirectCoboundary : std::false_type {};
    // UsesDenseUidIndex: the uid is a small dense integer, so uid->sorted_id is a flat
    //   direct-address array instead of a hash map (faster, smaller).
    template<class Cell>
    struct UsesDenseUidIndex : std::false_type {};

    constexpr size_t plus_inf = std::numeric_limits<size_t>::max();

    constexpr size_t k_invalid_index = std::numeric_limits<size_t>::max();
    constexpr dim_type k_all_dims = std::numeric_limits<dim_type>::max();

    // Stateless allocator routing through jemalloc (je_malloc/je_free) when the
    // build links it, else the system allocator. Empty, so it adds no size to a
    // container (empty-base optimization). Used both for reduction columns
    // (sparse_matrix.h) and for simplex/cell vertex vectors (IdxVector).
    template<class T>
    struct JeAllocator {
        using value_type = T;
        JeAllocator() noexcept = default;
        template<class U> JeAllocator(const JeAllocator<U>&) noexcept { }

        T* allocate(std::size_t n)
        {
#ifdef OINEUS_USE_JEMALLOC
            void* p = je_malloc(n * sizeof(T));
#else
            void* p = std::malloc(n * sizeof(T));
#endif
            if (p == nullptr)
                throw std::bad_alloc();
            return static_cast<T*>(p);
        }

        void deallocate(T* p, std::size_t) noexcept
        {
#ifdef OINEUS_USE_JEMALLOC
            je_free(p);
#else
            std::free(p);
#endif
        }

        template<class U> bool operator==(const JeAllocator<U>&) const noexcept { return true; }
        template<class U> bool operator!=(const JeAllocator<U>&) const noexcept { return false; }
    };

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

    // Number of bits needed to represent any id in [0, n): the bit-width of the
    // largest id n-1 (0 for n <= 1). Used to check that a dense slim-cell uid
    // (grid vertex id packed with face/type bits) fits the 63-bit budget of a
    // signed 64-bit oin_int before a grid filtration is built.
    inline int bits_for_count(std::size_t n)
    {
        if (n <= 1)
            return 0;
        int b = 0;
        std::size_t m = n - 1;
        while (m > 0) { m >>= 1; ++b; }
        return b;
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
