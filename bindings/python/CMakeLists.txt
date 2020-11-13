add_subdirectory(pybind11)

if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)        # to work with pip
    set (MODULE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else()
    set (MODULE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

file (MAKE_DIRECTORY "${MODULE_OUTPUT_DIRECTORY}/smilis")
file (GLOB SMILIS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/smilis/*.py")
add_custom_target           (smilis ALL ${CMAKE_COMMAND} -E copy_directory
                                       ${CMAKE_CURRENT_SOURCE_DIR}/smilis ${MODULE_OUTPUT_DIRECTORY}/smilis
                                       DEPENDS ${SMILIS_PYTHON})

pybind11_add_module         (_smilis smilis.cpp)
target_link_libraries       (_smilis PRIVATE ${libraries})
set_target_properties       (_smilis PROPERTIES OUTPUT_NAME smilis/_smilis)
