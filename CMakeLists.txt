cmake_minimum_required(VERSION 3.15)

project(oineus LANGUAGES CXX)

option(oin_use_spdlog "Use spdlog" OFF)
option(oin_use_jemalloc "Use jemalloc" ON)
option(oin_build_tests "Build tests" ON)
option(oin_build_examples "Build examples" ON)
option(oin_build_julia "Build Julia bindings (CxxWrap.jl)" OFF)
option(oin_gather_add_stats "Gather statistics about summand sizes" OFF)
option(oin_caliper "Enable profiling with Caliper" OFF)

# Default to Release

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package (Threads REQUIRED)
find_package (Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set (libraries ${libraries} ${CMAKE_THREAD_LIBS_INIT})

if (oin_use_jemalloc)
    # jemalloc is enabled by default: its thread-caching allocator roughly halves
    # the free-heavy copy-back phase of the parallel reduction (per-column frees +
    # cross-thread hazard-pointer reclamation). It is a HARD requirement here -- if
    # it is not found the configure step fails with the message below. Point CMake
    # at a non-standard install (e.g. spack) with -DJEMALLOC_ROOT=/prefix or the
    # JEMALLOC_ROOT env var; to fall back to the system allocator instead, pass
    # -Doin_use_jemalloc=OFF.
    find_library(JEMALLOC_LIBRARY NAMES jemalloc libjemalloc
            HINTS ${JEMALLOC_ROOT} ENV JEMALLOC_ROOT
                  /opt/homebrew/opt/jemalloc /opt/homebrew
                  /usr/local/opt/jemalloc /usr/local
            PATH_SUFFIXES lib lib64)
    if (JEMALLOC_LIBRARY)
        message(STATUS "oineus: using jemalloc: ${JEMALLOC_LIBRARY}")
        set (libraries ${libraries} ${JEMALLOC_LIBRARY})
        get_filename_component(JEMALLOC_LIBDIR "${JEMALLOC_LIBRARY}" DIRECTORY)
        list(APPEND CMAKE_BUILD_RPATH "${JEMALLOC_LIBDIR}")
        list(APPEND CMAKE_INSTALL_RPATH "${JEMALLOC_LIBDIR}")
    else()
        message(FATAL_ERROR
                "oineus: jemalloc not found, but oin_use_jemalloc is ON (the default).\n"
                "  jemalloc's thread-caching allocator roughly halves the parallel\n"
                "  reduction's copy-back phase, so it is required by default.\n"
                "  Fix this in one of these ways:\n"
                "    * install jemalloc: brew install jemalloc / apt install libjemalloc-dev /\n"
                "      spack install jemalloc;\n"
                "    * if it is installed in a non-standard prefix, pass -DJEMALLOC_ROOT=/prefix;\n"
                "    * to build WITHOUT jemalloc and use the system allocator instead, pass\n"
                "      -Doin_use_jemalloc=OFF.")
    endif()
endif()

if(oin_caliper)
    find_package(Caliper REQUIRED)
    add_compile_definitions (OINEUS_USE_CALIPER)
    set(libraries ${libraries} caliper)
endif()


if (oin_build_examples)
    add_subdirectory(examples)
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
