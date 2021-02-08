add_executable(tests tests_main.cpp tests_reduction.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2)
add_test(NAME all COMMAND tests)
