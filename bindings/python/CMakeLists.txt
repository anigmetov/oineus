add_subdirectory(pybind11)

if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)        # to work with pip
    set (MODULE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else()
    set (MODULE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

file (MAKE_DIRECTORY "${MODULE_OUTPUT_DIRECTORY}/oineus")
file (GLOB SMILIS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/oineus/*.py")
add_custom_target           (oineus ALL ${CMAKE_COMMAND} -E copy_directory
                                       ${CMAKE_CURRENT_SOURCE_DIR}/oineus ${MODULE_OUTPUT_DIRECTORY}/oineus
                                       DEPENDS ${SMILIS_PYTHON})

pybind11_add_module         (_oineus oineus.cpp)
target_link_libraries       (_oineus PRIVATE ${libraries})
set_target_properties       (_oineus PROPERTIES OUTPUT_NAME oineus/_oineus)
