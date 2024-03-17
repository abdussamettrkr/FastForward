cmake_minimum_required(VERSION 3.5.0)
project(tensor-cpp VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11)

include(CTest)
enable_testing()
add_executable(tensor-cpp main.cpp src/shape.cpp src/tensor.cpp src/binary_primitives.cpp src/utils.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp)
target_include_directories(tensor-cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(tensor_test tests/test_matmul.cpp src/shape.cpp src/tensor.cpp src/utils.cpp src/binary_primitives.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp)
target_include_directories(tensor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(broadcast_test tests/test_broadcast_shape.cpp src/utils.cpp src/tensor.cpp src/shape.cpp src/binary_primitives.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp)
target_include_directories(broadcast_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_test(
NAME tensor_test
COMMAND $<TARGET_FILE:tensor_test>
)

add_test(
NAME broadcast_test
COMMAND $<TARGET_FILE:broadcast_test>
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
