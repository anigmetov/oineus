project(oineus)

cmake_minimum_required(VERSION 3.4)

option      (use_spdlog     "Use spdlog"    off)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

if  (use_spdlog)
    add_definitions(-DARACHNE_USE_SPDLOG)
endif()

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -pedantic -Wextra -pedantic-errors")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -ggdb ")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")

set(PP_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/common_defs.h
               ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/diagram.h
               ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/filtration.h
               ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/includes.h
               ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/matrix.h
               ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/mem_reclamation.h
               ${CMAKE_CURRENT_SOURCE_DIR}/include/oineus/simplex.h
               )

set(PP_SRC_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)



find_package (Threads)
set (libraries ${libraries}   ${CMAKE_THREAD_LIBS_INIT})
set (libraries ${libraries}   jemalloc)

add_executable(oineus ${PP_SRC_FILES} ${PP_HEADERS})
target_link_libraries(oineus PUBLIC ${libraries} )
