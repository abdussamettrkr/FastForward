#include "utils.hpp"
#include "conv.hpp"

using namespace core;



bool test_im2col()
{
    bool result = false;
    const size_t inputHeight = 4;
    const size_t inputWidth = 4;
    const size_t inputChannels = 1;

    // Define kernel size
    const size_t kernelHeight = 2;
    const size_t kernelWidth = 2;

    // Define stride
    const size_t stride_h = 1;
    const size_t stride_w = 1;

    // Compute output dimensions
    const size_t outputHeight = (inputHeight - kernelHeight) / stride_h + 1;
    const size_t outputWidth = (inputWidth - kernelWidth) / stride_w + 1;

    // Define input image
    float input[inputHeight * inputWidth * inputChannels] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Allocate memory for expected output (im2col) matrix
    const size_t outputSize = outputHeight * outputWidth * kernelHeight * kernelWidth * inputChannels;
    float expectedOutput[outputSize] = {
        1, 2, 5, 6,
        2, 3, 6, 7,
        3, 4, 7, 8,
        5, 6, 9, 10,
        6, 7, 10, 11,
        7, 8, 11, 12,
        9, 10, 13, 14,
        10, 11, 14, 15,
        11, 12, 15, 16
    };

    // Allocate memory for output (im2col) matrix
    float* output = new float[outputSize];
    im2col(input, output, inputWidth, inputHeight, outputHeight, outputWidth, kernelHeight, kernelWidth, inputChannels, 4, 1);
    for (size_t i = 0; i < outputSize; ++i) {
        if(output[i] != expectedOutput[i]){
            result = true;     
        }
    }
    return result;
}

int main()
{
    return test_im2col();
}