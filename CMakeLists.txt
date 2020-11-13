project(oineus)

cmake_minimum_required(VERSION 3.4)

option(use_spdlog "Use spdlog" off)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 14)

find_package (Threads)
set (libraries ${libraries}   ${CMAKE_THREAD_LIBS_INIT})
set (libraries ${libraries}   jemalloc)

add_subdirectory(examples)
add_subdirectory(bindings/python)
