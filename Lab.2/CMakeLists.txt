cmake_minimum_required(VERSION 3.10)
project(MatrixMultOMP)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

add_executable(matrix_mult_omp matrix_mult_omp.cpp)

if(MSVC)
    target_link_libraries(matrix_mult_omp)
else()
    target_link_libraries(matrix_mult_omp ${OpenMP_CXX_LIBRARIES})
endif()