#include "unary_primitives.hpp"
#include "utils.hpp"


namespace core{
void Convolution::eval(const std::vector<core::Tensor>& inputs, core::Tensor& out){
    auto& input = inputs[0]; // N,H,W,C
    auto& kernel = inputs[1]; //O,H,W,C
    const std::vector<int> inShape = input.shape()->dims();
    const std::vector<int> kShape = kernel.shape()->dims();

    size_t N = inShape[0];
    size_t inH = inShape[1];
    size_t inW = inShape[2];
    size_t C = inShape[3];
    size_t O = kShape[0];
    size_t kH = kShape[1];
    size_t kW = kShape[2];

    const float* kData = kernel.data();
    const std::vector<int> kStrides = kernel.getStrides();
    const float* iData = input.data();
    const std::vector<int> iStrides = input.getStrides();

    float* outData = out.data();

    for (size_t n = 0; n < N; n++)
    {
        for (size_t h = 0; h < inH - (kH / 2) * 2; h++)
        {
            for (size_t w = 0; w < (inW / 2) * 2; w++)
            {
                for (size_t o = 0; o < O; o++)
                {
                    float res = 0;
                    for (size_t kh = 0; kh < kH; kh++)
                    {   
                        for (size_t kw = 0; kw < kW; kw++)
                        {

                            for (size_t c = 0; c < C; c++)
                            {
                                res += kData[o * kStrides[0] + kh * kStrides[1] + kw * kStrides[2] + c] *
                                 iData[n * iStrides[0] + h * iStrides[1] + w * iStrides[2] + c];
                            }
                        }
                    }
                    outData[n * iStrides[0] + h * iStrides[1] + w * iStrides[2] + o] = res;
                }
            }    
        }   
    }
}
}