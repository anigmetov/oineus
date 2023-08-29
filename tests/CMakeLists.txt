add_executable(tests tests_reduction.cpp tests_sparse_matrix.cpp)

target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                         ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

target_link_libraries(tests PRIVATE ${libraries})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME all COMMAND tests)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/a_6.txt"
           DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


# check if pytest module is installed
execute_process(COMMAND ${Python3_EXECUTABLE} -m pytest --version
                OUTPUT_VARIABLE PYTEST_output
                ERROR_VARIABLE  PYTEST_error
                RESULT_VARIABLE PYTEST_result)


execute_process(COMMAND ${Python3_EXECUTABLE} -c "import dionysus"
        OUTPUT_VARIABLE DION_output
        ERROR_VARIABLE  DION_error
        RESULT_VARIABLE DION_result)


if(NOT ${PYTEST_result} EQUAL 0)
    message(WARNING "Pytest package not available: executable ${Python3_EXECUTABLE}, ${PYTEST_error}, Python tests will be ignored. Use pip install [--user] pytest to run them.")
else()
#    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test_oin_vertebra.py"
#            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

    file(GLOB VERTEBRA_NPY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*vertebra*.npy")
    file(GLOB TEST_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

    file(COPY ${VERTEBRA_NPY_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    file(COPY ${TEST_PY_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

#    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test_oin_random.py"
#            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
#
#    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/kernel-pytest.py"

    add_test(NAME induced-matching-pytest
            COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/induced-matching-pytest.py" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME kernel-pytest
            COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/kernel-pytest.py" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME oin-vertebra-pytest
            COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_oin_vertebra.py" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    if(${DION_result} EQUAL 0)
        add_test(NAME oin-random-pytest
                COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_oin_random.py" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
endif()
