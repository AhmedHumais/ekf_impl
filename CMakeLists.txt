cmake_minimum_required(VERSION 2.8.3)

project(ekf_impl)

include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/Eigen
)

add_library(${PROJECT_NAME} src/ekf_impl.cpp)
target_link_libraries(${PROJECT_NAME})

add_executable(test_ekf src/main.cpp)
target_link_libraries(test_ekf ${PROJECT_NAME})