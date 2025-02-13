cmake_minimum_required(VERSION 3.14)

project(oineus LANGUAGES CXX)

option(oin_use_spdlog "Use spdlog" OFF)
option(oin_use_jemalloc "Use jemalloc" OFF)
option(oin_build_tests "Build tests" ON)
option(oin_build_examples "Build examples" ON)
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
    set (libraries ${libraries} jemalloc)
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

if (oin_build_tests)
    add_subdirectory(extern/Catch2)
    enable_testing()
    add_subdirectory(tests)
endif()
