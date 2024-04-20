#include "utils.hpp"
#include "ops.hpp"

using namespace core;

bool test_im2col(const core::Tensor &input, const float expected[], size_t kH, size_t kW, size_t stride)
{

    auto inShape = input.shape();

    const size_t outH = (inShape[1] - kH) / stride + 1;
    const size_t outW = (inShape[2] - kW) / stride + 1;
    auto result = ops::im2col(input, kH, kW, 0, stride);
    auto resultData = result.data();

    const size_t outputSize = outH * outW * kH * kW * inShape[3];

    for (size_t i = 0; i < outputSize; ++i)
    {
        if (resultData[i] != expected[i])
        {
            return true;
        }
    }
    return false;
}

bool im2col_test_1()
{
    float inputData[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};

    float expectedOutput[] = {
        1, 2, 5, 6,
        2, 3, 6, 7,
        3, 4, 7, 8,
        5, 6, 9, 10,
        6, 7, 10, 11,
        7, 8, 11, 12,
        9, 10, 13, 14,
        10, 11, 14, 15,
        11, 12, 15, 16};

    // Create input tensor
    auto input = core::Tensor({1, 4, 4, 1}, {16, 4, 1, 1}, inputData);

    return test_im2col(input, expectedOutput, 2, 2, 1);
}

bool im2col_test_2()
{

    float inputData[] = {
        1, 2, 3, 4, 17, 18, 19, 20,
        5, 6, 7, 8, 21, 22, 23, 24,
        9, 10, 11, 12, 25, 26, 27, 28,
        13, 14, 15, 16, 29, 30, 31, 32};
    float expectedOutput[] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        3, 4, 17, 18, 7, 8, 21, 22,
        17, 18, 19, 20, 21, 22, 23, 24,
        5, 6, 7, 8, 9, 10, 11, 12,
        7, 8, 21, 22, 11, 12, 25, 26,
        21, 22, 23, 24, 25, 26, 27, 28,
        9, 10, 11, 12, 13, 14, 15, 16,
        11, 12, 25, 26, 15, 16, 29, 30,
        25, 26, 27, 28, 29, 30, 31, 32};

    // Create input tensor
    auto input = core::Tensor({1, 4, 4, 2}, {32, 8, 2, 1}, inputData);

    return test_im2col(input, expectedOutput, 2, 2, 1);
}

int main()
{
    bool res1 = im2col_test_1();
    bool res2 = im2col_test_2();
    return res1 | res2;
}