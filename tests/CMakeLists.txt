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

    add_test(NAME py-bindings-cells
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_bindings_cells_api.py")
    set_tests_properties(py-bindings-cells PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-bindings-common
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_bindings_common_api.py")
    set_tests_properties(py-bindings-common PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-bindings-decomposition
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_bindings_decomposition_api.py" -k "not test_decomposition_diagram_overloads_cube_1d")
    set_tests_properties(py-bindings-decomposition PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-restore-elz
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_restore_elz_parallel.py")
    set_tests_properties(py-restore-elz PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-bindings-diagram
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_bindings_diagram_api.py")
    set_tests_properties(py-bindings-diagram PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-bindings-filtration
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_bindings_filtration_api.py")
    set_tests_properties(py-bindings-filtration PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-bindings-kicr
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_bindings_kicr_api.py")
    set_tests_properties(py-bindings-kicr PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-cube
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_cube.py")
    set_tests_properties(py-cube PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-cubical-dgms
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_cubical_dgms.py")
    set_tests_properties(py-cubical-dgms PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-induced-matching
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/induced-matching-pytest.py")
    set_tests_properties(py-induced-matching PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-kernel
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/kernel-pytest.py")
    set_tests_properties(py-kernel PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: enable after updating to current KICR API.
    add_test(NAME py-kernel-legacy
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/kernel-test-1.py")
    set_tests_properties(py-kernel-legacy PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: this file is a CLI script and not a pytest module yet.
    # add_test(NAME py-oin
    #         COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_oin.py")
    # set_tests_properties(py-oin PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-oin-random
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_oin_random.py")
    set_tests_properties(py-oin-random PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-oin-vertebra
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_oin_vertebra.py")
    set_tests_properties(py-oin-vertebra PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: this file has no pytest tests yet (script-style debugging helper).
    # add_test(NAME py-kicr-legacy-1
    #         COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_kicr_1.py")
    # set_tests_properties(py-kicr-legacy-1 PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: this file has no pytest tests yet (script-style debugging helper).
    # add_test(NAME py-kicr-legacy-2
    #         COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_kicr_2.py")
    # set_tests_properties(py-kicr-legacy-2 PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-pickle
            COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_BINARY_DIR}/test_pickle.py")
    set_tests_properties(py-pickle PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-cone
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_cone.py")
    set_tests_properties(py-example-cone PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: enable after fixing API issues in this example script.
    # add_test(NAME py-example-cone-diff
    #         COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_cone_diff.py")
    # set_tests_properties(py-example-cone-diff PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-cube
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_cube.py")
    set_tests_properties(py-example-cube PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-diff-cyl
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_diff_cyl.py")
    set_tests_properties(py-example-diff-cyl PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-diff-vr-dists
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_diff_vr_dists.py")
    set_tests_properties(py-example-diff-vr-dists PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-diff-vr-pts
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_diff_vr_pts.py")
    set_tests_properties(py-example-diff-vr-pts PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: enable after updating to the current kernel/cokernel API.
    # add_test(NAME py-example-kernel
    #         COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_kernel.py")
    # set_tests_properties(py-example-kernel PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-kernel-cyl
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_kernel_cyl.py")
    set_tests_properties(py-example-kernel-cyl PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-manual
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_manual.py")
    set_tests_properties(py-example-manual PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-opt
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_opt.py")
    set_tests_properties(py-example-opt PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-opt-vr
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_opt_vr.py")
    set_tests_properties(py-example-opt-vr PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-vr
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_vr.py")
    set_tests_properties(py-example-vr PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-example-wasserstein-opt
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_example_wasserstein_opt.py")
    set_tests_properties(py-example-wasserstein-opt PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_test(NAME py-differentiable-diagram
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/test_differentiable_diagram.py")
    set_tests_properties(py-differentiable-diagram PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    add_custom_target(py-tests
            COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure -R "^py-")
endif()
