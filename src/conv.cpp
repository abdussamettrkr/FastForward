#include "unary_primitives.hpp"
#include "utils.hpp"


namespace core{
void Convolution::eval(const std::vector<core::Tensor>& inputs, core::Tensor& out){
    auto& input = inputs[0]; // N,H,W,C
    auto& kernel = inputs[1]; //O,H,W,C
    const std::vector<int> outShape = out.shape();
    const std::vector<int> kShape = kernel.shape();

    size_t N = outShape[0];
    size_t inH = outShape[1];
    size_t inW = outShape[2];
    size_t C = kShape[3];
    size_t O = kShape[0];
    size_t kH = kShape[1];
    size_t kW = kShape[2];

    const float* kData = kernel.data();
    const std::vector<int> kStrides = kernel.strides();
    const float* iData = input.data();
    const std::vector<int> iStrides = input.strides();
    const std::vector<int> outStrides = out.strides();

    float* outData = out.data();

    for (size_t n = 0; n < N; n++)
    {
        for (size_t h = 0; h < inH; h++)
        {
            for (size_t w = 0; w < inW; w++)
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
                                iData[n * iStrides[0] + (h * stride + kh) * iStrides[1] + (w * stride + kw) * iStrides[2] + c];
                            }
                        }
                    }
                    outData[n * outStrides[0] + h * outStrides[1] + w * outStrides[2] + o] = res;
                }
            }    
        }   
    }
}
}