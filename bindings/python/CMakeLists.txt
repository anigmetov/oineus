add_subdirectory(pybind11)

if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)        # to work with pip
    set (MODULE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else()
    set (MODULE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

file (MAKE_DIRECTORY "${MODULE_OUTPUT_DIRECTORY}/oineus")
file (GLOB OINEUS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/oineus/*.py")

add_custom_target           (oineus ALL ${CMAKE_COMMAND} -E copy_directory
                                       ${CMAKE_CURRENT_SOURCE_DIR}/oineus ${MODULE_OUTPUT_DIRECTORY}/oineus
                                       DEPENDS ${OINEUS_PYTHON})

set(OINEUS_PYTHON_INT "long int" CACHE STRING "Choose the Int type in Python bindings, options are: int, long int, long long int." FORCE)
set_property(CACHE OINEUS_PYTHON_INT PROPERTY STRINGS "int" "long int" "long long int")

set(OINEUS_PYTHON_REAL "double" CACHE STRING "Choose the Real type in Python bindings, options are: float, double." FORCE)
set_property(CACHE OINEUS_PYTHON_REAL PROPERTY STRINGS "float" "double")

pybind11_add_module         (_oineus oineus.cpp
                                     oineus_common.cpp
                                     oineus_decomposition.cpp
                                     oineus_diagram.cpp
                                     oineus_cells.cpp
                                     oineus_filtration.cpp
                                     oineus_kicr.cpp
                                     oineus_functions.cpp
                                     oineus_top_optimizer.cpp
                            )
target_compile_definitions(_oineus PRIVATE OINEUS_PYTHON_REAL=${OINEUS_PYTHON_REAL} OINEUS_PYTHON_INT=${OINEUS_PYTHON_INT})
target_link_libraries       (_oineus PRIVATE ${libraries})
target_include_directories  (_oineus PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
                                              "${CMAKE_CURRENT_SOURCE_DIR}/../../extern")
set_target_properties       (_oineus PROPERTIES OUTPUT_NAME oineus/_oineus)

if (oin_use_spdlog)
    target_compile_definitions(_oineus PRIVATE -DOINEUS_USE_SPDLOG)
endif()

if (oin_gather_add_stats)
    target_compile_definitions(_oineus PRIVATE -DOINEUS_GATHER_ADD_STATS)
endif()

if(oin_build_tests)
    add_custom_target       (oineus_copy_to_tests ALL
            ${CMAKE_COMMAND} -E copy_directory ${MODULE_OUTPUT_DIRECTORY}/oineus "${CMAKE_CURRENT_BINARY_DIR}/../../tests/oineus"
            DEPENDS _oineus oineus)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_oineus PRIVATE -Wall -Wextra -Wpedantic)
endif()
