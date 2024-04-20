cmake_minimum_required(VERSION 3.5.0)
project(tensor-cpp VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11)

include(CTest)
enable_testing()
set(CMAKE_CXX_FLAGS "-O3")
add_executable(tensor-cpp main.cpp src/tensor.cpp src/binary_primitives.cpp src/utils.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp src/maxpool.cpp  src/copy.cpp)
target_include_directories(tensor-cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(alexnet examples/alexnet.cpp src/tensor.cpp src/binary_primitives.cpp src/utils.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp src/maxpool.cpp  src/copy.cpp)
target_include_directories(alexnet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(bench_conv2d benchmark/benchmark_conv2d.cpp src/tensor.cpp src/binary_primitives.cpp src/utils.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp src/maxpool.cpp  src/copy.cpp)
target_include_directories(bench_conv2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(conv2dtest tests/test_conv2d.cpp src/tensor.cpp src/utils.cpp src/binary_primitives.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp src/maxpool.cpp  src/copy.cpp)
target_include_directories(conv2dtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(matmultest tests/test_matmul.cpp src/tensor.cpp src/utils.cpp src/binary_primitives.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp src/maxpool.cpp  src/copy.cpp)
target_include_directories(matmultest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# add_executable(broadcast_test tests/test_broadcast_shape.cpp src/utils.cpp src/tensor.cpp src/binary_primitives.cpp src/unary_primitives.cpp src/matmul.cpp src/conv.cpp src/ops.cpp src/maxpool.cpp  src/copy.cpp)
# target_include_directories(broadcast_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_test(
NAME matmultest
COMMAND $<TARGET_FILE:matmultest>
)

add_test(
NAME conv2dtest
COMMAND $<TARGET_FILE:conv2dtest>
)

# add_test(
# NAME broadcast_test
# COMMAND $<TARGET_FILE:broadcast_test>
# )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
