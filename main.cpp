#include "ops.hpp"

int main()
{
    core::Tensor input = core::Tensor::ones({2, 128, 128, 3});
    float * data = input.data();
    data[0] = 3.5;
    data[1] = 0.2;
    core::Tensor kernel = core::Tensor::ones({32, 2, 2, 3});
    core::Tensor kernelFlat = kernel.flatten(1, 3);
    core::Tensor colImg = ops::im2col(input, 2, 2, 0, 1);
    auto result1 = ops::matmul(colImg, kernelFlat, true);
    auto result = ops::conv2d(input, kernel);
}