foreach(name oineus_lower_star oineus_v_opt oineus_vectorizer oineus_test_debug)
    add_executable(${name} "${name}.cpp")

    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                                ${CMAKE_CURRENT_SOURCE_DIR}/../extern
                                                ${CMAKE_CURRENT_SOURCE_DIR}/../extern/opts/include)

    if (oin_use_spdlog)
        target_compile_definitions(${name} PRIVATE -DOINEUS_USE_SPDLOG)
    endif()

    target_link_libraries(${name} PUBLIC ${libraries})
endforeach()
