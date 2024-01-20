cmake_minimum_required(VERSION 3.0.0)
project(tensor-cpp VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11)

include(CTest)
enable_testing()
add_executable(tensor-cpp main.cpp src/shape.cpp src/tensor.cpp)
target_include_directories(tensor-cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(tensor_test tests/test_matmul.cpp src/shape.cpp src/tensor.cpp)
target_include_directories(tensor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_test(
  NAME tensor_test
  COMMAND $<TARGET_FILE:tensor_test>
  )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
