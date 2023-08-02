if (PVFMM_ENABLE_CUDA)
  find_package(CUDAToolkit)
  set(example_DEPS CUDA::cublas_static CUDA::cublasLt_static)
else()
  unset(example_DEPS)
endif()


add_executable(example-c ./src/example-c.c)
target_link_libraries(example-c PRIVATE pvfmmStatic "${example_DEPS}")
target_include_directories(example-c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(example2-c ./src/example2-c.c)
target_link_libraries(example2-c PRIVATE pvfmmStatic "${example_DEPS}")
target_include_directories(example2-c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_executable(example1 ./src/example1.cpp)
target_link_libraries(example1 PRIVATE pvfmmStatic "${example_DEPS}")
target_include_directories(example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(example2 ./src/example2.cpp)
target_link_libraries(example2 PRIVATE pvfmmStatic "${example_DEPS}")
target_include_directories(example2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(fmm_pts ./src/fmm_pts.cpp)
target_link_libraries(fmm_pts PRIVATE pvfmmStatic "${example_DEPS}")
target_include_directories(fmm_pts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(fmm_cheb ./src/example1.cpp)
target_link_libraries(fmm_cheb PRIVATE pvfmmStatic "${example_DEPS}")
target_include_directories(fmm_cheb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
