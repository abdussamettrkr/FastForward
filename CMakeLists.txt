cmake_minimum_required(VERSION 3.0.0)
project(tensor-cpp VERSION 0.1.0)

include(CTest)
enable_testing()
add_executable(tensor-cpp main.cpp src/shape.cpp src/tensor.cpp)
target_include_directories(tensor-cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
