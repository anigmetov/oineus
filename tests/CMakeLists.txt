add_executable(tests tests_reduction.cpp tests_sparse_matrix.cpp)

target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                         ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

target_link_libraries(tests PRIVATE ${libraries})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME all COMMAND tests)

add_executable(tests_parallel_col_to_row tests_parallel_col_to_row.cpp)

target_include_directories(tests_parallel_col_to_row PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                                             ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

target_link_libraries(tests_parallel_col_to_row PRIVATE ${libraries})
target_link_libraries(tests_parallel_col_to_row PRIVATE Catch2::Catch2WithMain)

add_test(NAME parallel-col-to-row COMMAND tests_parallel_col_to_row)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/a_6.txt"
           DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


# check if pytest module is installed
execute_process(COMMAND ${Python_EXECUTABLE} -m pytest --version
                OUTPUT_VARIABLE PYTEST_output
                ERROR_VARIABLE  PYTEST_error
                RESULT_VARIABLE PYTEST_result)


if(NOT ${PYTEST_result} EQUAL 0)
    message(WARNING "Pytest package not available: executable ${Python_EXECUTABLE}, ${PYTEST_error}, Python tests will be ignored. Use pip install [--user] pytest to run them.")
else()
    file(GLOB VERTEBRA_NPY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*vertebra*.npy")
    file(GLOB TEST_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

    file(COPY ${VERTEBRA_NPY_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    file(COPY ${TEST_PY_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

    # Keep copied Python tests/data in sync on every build (without reconfigure).
    set(COPY_PY_TEST_FILES_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/copy_py_test_files.cmake")
    file(WRITE "${COPY_PY_TEST_FILES_SCRIPT}" "file(GLOB VERTEBRA_NPY_FILES \"${CMAKE_CURRENT_SOURCE_DIR}/*vertebra*.npy\")\n")
    file(APPEND "${COPY_PY_TEST_FILES_SCRIPT}" "file(GLOB TEST_PY_FILES \"${CMAKE_CURRENT_SOURCE_DIR}/*.py\")\n")
    file(APPEND "${COPY_PY_TEST_FILES_SCRIPT}" "foreach(src IN LISTS VERTEBRA_NPY_FILES TEST_PY_FILES)\n")
    file(APPEND "${COPY_PY_TEST_FILES_SCRIPT}" "  file(COPY \"\${src}\" DESTINATION \"${CMAKE_CURRENT_BINARY_DIR}\")\n")
    file(APPEND "${COPY_PY_TEST_FILES_SCRIPT}" "endforeach()\n")

    add_custom_target(copy_py_test_files ALL
            COMMAND ${CMAKE_COMMAND} -P "${COPY_PY_TEST_FILES_SCRIPT}")
    add_dependencies(tests copy_py_test_files)
    add_dependencies(tests_parallel_col_to_row copy_py_test_files)

    # Convention: CTest case <foo-bar> runs ${CMAKE_CURRENT_BINARY_DIR}/test_<foo_bar>.py
    # (dashes in the test name become underscores in the file name).

    # Pytest-driven tests.
    foreach(NAME IN ITEMS
            api-cells
            api-common
            api-dcmp
            api-dgm
            api-fil
            api-kicr
            bot-match
            cube
            cube-dgm
            dcmp-restore-elz
            dgm-dist
            diff-critical-sets
            diff-cube-fil
            diff-grad
            diff-wass
            fr-dgm-random
            fr-dgm-vertebra
            frechet
            ind-match
            kicr
            match-rv-policy
            pickle
            plot
            plot-chain
            plot-gradient
            sliced-wass
            wass-match)
        string(REPLACE "-" "_" STEM "${NAME}")
        add_test(NAME py-${NAME}
                COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_${STEM}.py")
        set_tests_properties(py-${NAME} PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endforeach()

    # Script-style example tests (no pytest discovery, just run the file).
    foreach(NAME IN ITEMS
            example-diff-cube
            example-diff-cyl
            example-kicr-cyl
            example-opt-fr
            example-opt-vr
            example-opt-wass)
        string(REPLACE "-" "_" STEM "${NAME}")
        add_test(NAME py-${NAME}
                COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_${STEM}.py")
        set_tests_properties(py-${NAME} PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endforeach()

    # TODO: enable after fixing API issues in this example script.
    # add_test(NAME py-example-cone-diff
    #         COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_cone_diff.py")
    # set_tests_properties(py-example-cone-diff PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: enable after updating to the current kernel/cokernel API.
    # add_test(NAME py-example-kernel
    #         COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_kernel.py")
    # set_tests_properties(py-example-kernel PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_custom_target(py-tests
            COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure -R "^py-")
endif()
