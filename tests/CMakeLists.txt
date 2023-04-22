add_executable(tests tests_reduction.cpp tests_sparse_matrix.cpp)

target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                         ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

target_link_libraries(tests PRIVATE ${libraries})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME all COMMAND tests)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/a_6.txt"
           DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


# check if pytest module is installed
execute_process(COMMAND python3 -m pytest --version
                OUTPUT_VARIABLE PYTEST_output
                ERROR_VARIABLE  PYTEST_error
                RESULT_VARIABLE PYTEST_result)

if(NOT ${PYTEST_result} EQUAL 0)
    message(WARNING "Pytest package not available: ${PYTEST_error}, Python tests will be ignored. Use pip install [--user] pytest to run them.")
else()
    # run python tests
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/kernel-pytest.py"
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME kernel-pytest
            COMMAND ${PYTHON_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/kernel-pytest.py" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

