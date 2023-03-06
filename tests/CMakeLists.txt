add_executable(tests tests_reduction.cpp tests_sparse_matrix.cpp)

target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                         ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

target_link_libraries(tests PRIVATE ${libraries})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME all COMMAND tests)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/a_6.txt"
           DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
