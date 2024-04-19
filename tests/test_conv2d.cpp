#include "utils.hpp"
#include "ops.hpp"

using namespace core;

bool test_im2col()
{
    const size_t inH = 4;
    const size_t inW = 4;
    const size_t inC = 1;

    const size_t kH = 2;
    const size_t kW = 2;

    const size_t stride_h = 1;
    const size_t stride_w = 1;
    const size_t outH = (inH - kH) / stride_h + 1;
    const size_t outW = (inW - kW) / stride_w + 1;

    float inputData[inH * inW * inC] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};

    auto input = core::Tensor({1, inH, inW, inC}, {inC * inW * inH, inC * inW, inC, 1}, inputData);

    auto result = ops::im2col(input, kH, kW, 0, 1);

    const size_t outputSize = outH * outW * kH * kW * inC;
    float expectedOutput[outputSize] = {
        1, 2, 5, 6,
        2, 3, 6, 7,
        3, 4, 7, 8,
        5, 6, 9, 10,
        6, 7, 10, 11,
        7, 8, 11, 12,
        9, 10, 13, 14,
        10, 11, 14, 15,
        11, 12, 15, 16};

    float *output = result.data();
    for (size_t i = 0; i < outputSize; ++i)
    {
        if (output[i] != expectedOutput[i])
        {
            return true;
        }
    }
    return false;
}

int main()
{
    return test_im2col();
}