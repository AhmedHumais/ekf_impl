cmake_minimum_required(VERSION 3.9.1)

project(ekf_impl)

find_package(OpenMP)

include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/Eigen
)

add_compile_options(-O3 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG -std=c++11 -march=native )

add_library(${PROJECT_NAME} src/ekf_impl.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

add_executable(test_ekf src/main.cpp)
target_link_libraries(test_ekf ${PROJECT_NAME})