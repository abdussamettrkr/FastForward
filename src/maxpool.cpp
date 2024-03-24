#include "unary_primitives.hpp"
#include "utils.hpp"
#include <limits>


namespace core{
void MaxPool2D::eval(const std::vector<core::Tensor>& inputs, core::Tensor& out){
    auto& input = inputs[0]; // N,H,W,C
    const std::vector<int> outShape = out.shape();
    
    size_t N = outShape[0];
    size_t outH = outShape[1];
    size_t outW = outShape[2];
    size_t C = outShape[3];

    const float* iData = input.data();
    float* outData = out.data();
    
    const std::vector<int> inStrides = input.strides();
    const std::vector<int> outStrides = out.strides();


    for (size_t n = 0; n < N; n++)
    {
        for (size_t h = 0; h < outH; h++)
        {
            for (size_t w = 0; w < outW; w++)
            {
                for (size_t c = 0; c < C; c++)
                {
                    float res = std::numeric_limits<float>::lowest();
                    for (size_t kh = 0; kh < kernel_size; kh++)
                    {   
                        for (size_t kw = 0; kw < kernel_size; kw++)
                        {
                            res = std::max(res, iData[n * inStrides[0] + (h * stride + kh) * inStrides[1] + (w * stride + kw) * inStrides[2] + C]);
                        }
                    }
                    outData[n * outStrides[0] + h * outStrides[1] + w * outStrides[2] + c] = res;
                }
            }    
        }   
    }
}
}