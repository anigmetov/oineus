cmake_minimum_required(VERSION 3.15)

project(oineus LANGUAGES CXX)

option(oin_use_spdlog "Use spdlog" OFF)
option(oin_use_jemalloc "Use jemalloc" ON)
option(oin_build_tests "Build tests" ON)
option(oin_build_examples "Build examples" ON)
option(oin_build_benchmarks "Build benchmarks" OFF)
option(oin_build_julia "Build Julia bindings (CxxWrap.jl)" OFF)
option(oin_gather_add_stats "Gather statistics about summand sizes" OFF)
option(oin_caliper "Enable profiling with Caliper" OFF)

# Default to Release

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The vendored icecream debug-print library is not C++20-clean (its char-type
# Tree handling is ambiguous under std::construct_at). All library IC() calls are
# debug no-ops behind this guard, so disable icecream globally on C++20.
add_compile_definitions(OINEUS_DISABLE_ICECREAM)

find_package (Threads REQUIRED)
find_package (Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set (libraries ${libraries} ${CMAKE_THREAD_LIBS_INIT})

if (oin_use_jemalloc)
    # jemalloc is VENDORED (extern/jemalloc) and statically linked, so it ships
    # inside the wheel with no runtime dependency. It is built with a symbol prefix
    # (je_malloc/je_free, no plain malloc/free, no interposition), and the reduction
    # column types call je_malloc/je_free explicitly (the OINEUS_USE_JEMALLOC paths
    # in sparse_matrix.h), so the allocator is actually used on every platform --
    # not merely where it happens to interpose. Its thread-caching allocator roughly
    # halves the free-heavy copy-back phase of the parallel reduction. To build
    # WITHOUT it (system allocator), pass -Doin_use_jemalloc=OFF.
    # Build the vendored jemalloc at CONFIGURE time (guarded -> one-time), so its
    # prefixed header is installed before any TU that includes sparse_matrix.h is
    # compiled. (An ExternalProject build races the targets under -j and the wrong
    # jemalloc.h -- e.g. an unprefixed Homebrew one pulled in via Boost's include
    # dir -- can shadow ours.)
    set(JEMALLOC_PREFIX   "${CMAKE_BINARY_DIR}/jemalloc")
    set(JEMALLOC_BUILDDIR "${CMAKE_BINARY_DIR}/jemalloc-build")
    set(JEMALLOC_INCLUDE  "${JEMALLOC_PREFIX}/include")
    set(JEMALLOC_STATIC   "${JEMALLOC_PREFIX}/lib/libjemalloc.a")
    if (NOT EXISTS "${JEMALLOC_STATIC}")
        message(STATUS "oineus: building vendored jemalloc (prefixed, static PIC) -- one-time...")
        file(MAKE_DIRECTORY "${JEMALLOC_BUILDDIR}")
        execute_process(
                COMMAND "${CMAKE_SOURCE_DIR}/extern/jemalloc/configure"
                        "--prefix=${JEMALLOC_PREFIX}" "--with-jemalloc-prefix=je_"
                        "--disable-cxx" "--disable-shared" "--enable-static" "EXTRA_CFLAGS=-fPIC"
                WORKING_DIRECTORY "${JEMALLOC_BUILDDIR}"
                RESULT_VARIABLE jem_cfg
                OUTPUT_FILE "${JEMALLOC_BUILDDIR}/oin_configure.log"
                ERROR_FILE  "${JEMALLOC_BUILDDIR}/oin_configure.log")
        if (NOT jem_cfg EQUAL 0)
            message(FATAL_ERROR "oineus: jemalloc configure failed (see "
                    "${JEMALLOC_BUILDDIR}/oin_configure.log). Pass -Doin_use_jemalloc=OFF "
                    "to build with the system allocator instead.")
        endif()
        execute_process(
                COMMAND make build_lib_static install_lib_static install_include
                WORKING_DIRECTORY "${JEMALLOC_BUILDDIR}"
                RESULT_VARIABLE jem_make
                OUTPUT_FILE "${JEMALLOC_BUILDDIR}/oin_make.log"
                ERROR_FILE  "${JEMALLOC_BUILDDIR}/oin_make.log")
        if (NOT jem_make EQUAL 0 OR NOT EXISTS "${JEMALLOC_STATIC}")
            message(FATAL_ERROR "oineus: jemalloc build failed (see "
                    "${JEMALLOC_BUILDDIR}/oin_make.log). Pass -Doin_use_jemalloc=OFF.")
        endif()
    endif()
    message(STATUS "oineus: using vendored jemalloc: ${JEMALLOC_STATIC}")
    # Our prefixed header must win over any system/Homebrew jemalloc.h on the path.
    include_directories(BEFORE "${JEMALLOC_INCLUDE}")
    set (libraries ${libraries} "${JEMALLOC_STATIC}" ${CMAKE_DL_LIBS})
    add_compile_definitions(OINEUS_USE_JEMALLOC)
endif()

if(oin_caliper)
    find_package(Caliper REQUIRED)
    add_compile_definitions (OINEUS_USE_CALIPER)
    set(libraries ${libraries} caliper)
endif()


if (oin_build_examples)
    add_subdirectory(examples)
endif ()

if (oin_build_benchmarks)
    add_subdirectory(benchmarks)
endif ()

add_subdirectory(bindings/python)
if (oin_build_julia)
    add_subdirectory(bindings/julia)
endif()

if (oin_build_tests)
    add_subdirectory(extern/Catch2)
    enable_testing()
    add_subdirectory(tests)
endif()
