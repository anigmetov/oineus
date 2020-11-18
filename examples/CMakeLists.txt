set(OINEUS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/common_defs.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/diagram.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/filtration.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/lower_star.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/includes.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/matrix.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/mem_reclamation.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/../include/oineus/simplex.h)

add_executable(oineus_lower_star oineus_lower_star.cpp ${PP_HEADERS} ../include/oineus/params.h)

target_compile_options(oineus_lower_star PRIVATE -Wall -pedantic -Wextra -pedantic-errors)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(oineus_lower_star PRIVATE -g -ggdb)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(oineus_lower_star PRIVATE -g -O3)
endif()

target_include_directories(oineus_lower_star PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                             ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

if (use_spdlog)
    target_compile_definitions(oineus_lower_star PRIVATE -DOINEUS_USE_SPDLOG)
endif()

target_link_libraries(oineus_lower_star PUBLIC ${libraries} )
