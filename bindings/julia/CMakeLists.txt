find_package(JlCxx CONFIG REQUIRED)
find_program(JULIA_EXECUTABLE NAMES julia REQUIRED)

add_library(oineus_julia SHARED oineus_julia.cpp)

target_link_libraries(oineus_julia
    PRIVATE
        JlCxx::cxxwrap_julia
        JlCxx::cxxwrap_julia_stl
        ${libraries}
)

target_include_directories(oineus_julia
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../extern"
)

if (oin_use_spdlog)
    target_compile_definitions(oineus_julia PRIVATE -DOINEUS_USE_SPDLOG)
endif()

if (oin_gather_add_stats)
    target_compile_definitions(oineus_julia PRIVATE -DOINEUS_GATHER_ADD_STATS)
endif()

# IceCream currently fails to compile on Apple Clang/libc++ in this TU.
target_compile_definitions(oineus_julia PRIVATE -DOINEUS_DISABLE_ICECREAM)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(oineus_julia PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(oineus_julia PROPERTIES
    OUTPUT_NAME oineus_julia
)

install(TARGETS oineus_julia
    LIBRARY DESTINATION bindings/julia/Oineus.jl/lib
    ARCHIVE DESTINATION bindings/julia/Oineus.jl/lib
    RUNTIME DESTINATION bindings/julia/Oineus.jl/lib
)

enable_testing()
add_test(
    NAME julia-bindings
    COMMAND
        ${JULIA_EXECUTABLE}
        --compiled-modules=no
        ${CMAKE_CURRENT_SOURCE_DIR}/Oineus.jl/test/run_tests.jl
)
set_tests_properties(
    julia-bindings
    PROPERTIES
        ENVIRONMENT "OINEUS_JULIA_LIBRARY=$<TARGET_FILE:oineus_julia>"
)
