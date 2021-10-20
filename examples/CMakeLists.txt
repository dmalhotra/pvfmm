add_executable(example1 ./src/example1.cpp)
target_link_libraries(example1 PRIVATE pvfmmStatic)
target_include_directories(example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(example2 ./src/example2.cpp)
target_link_libraries(example2 PRIVATE pvfmmStatic)
target_include_directories(example2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(fmm_pts ./src/fmm_pts.cpp)
target_link_libraries(fmm_pts PRIVATE pvfmmStatic)
target_include_directories(fmm_pts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(fmm_cheb ./src/example1.cpp)
target_link_libraries(fmm_cheb PRIVATE pvfmmStatic)
target_include_directories(fmm_cheb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_new_kernels ./src/test_new_kernels.cpp)
target_link_libraries(test_new_kernels PRIVATE pvfmmStatic)
target_include_directories(test_new_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
