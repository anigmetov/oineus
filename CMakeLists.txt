cmake_minimum_required(VERSION 3.5)

project(oineus LANGUAGES CXX)

option(use_spdlog "Use spdlog" OFF)
option(use_jemalloc "Use jemalloc" OFF)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)

find_package (Threads REQUIRED)
find_package (TBB REQUIRED)
find_package (Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set (libraries ${libraries} ${CMAKE_THREAD_LIBS_INIT} TBB::tbb)

if (use_jemalloc)
    set (libraries ${libraries} jemalloc)
endif()

add_subdirectory(examples)
add_subdirectory(bindings/python)

find_package(Catch2)
if (Catch2_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()

