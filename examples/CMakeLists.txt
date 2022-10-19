add_executable(oineus_lower_star oineus_lower_star.cpp)
add_executable(oineus_v_opt oineus_v_opt.cpp)
add_executable(oineus_vectorizer oineus_vectorizer.cpp)

target_include_directories(oineus_lower_star PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

target_include_directories(oineus_v_opt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/../extern)


target_include_directories(oineus_vectorizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/../extern)

if (use_spdlog)
    target_compile_definitions(oineus_lower_star PRIVATE -DOINEUS_USE_SPDLOG)
    target_compile_definitions(oineus_v_opt PRIVATE -DOINEUS_USE_SPDLOG)
    target_compile_definitions(oineus_vectorizer PRIVATE -DOINEUS_USE_SPDLOG)
endif()

target_link_libraries(oineus_lower_star PUBLIC ${libraries})
target_link_libraries(oineus_v_opt PUBLIC ${libraries})
target_link_libraries(oineus_vectorizer PUBLIC ${libraries})
