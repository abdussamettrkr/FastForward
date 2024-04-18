#include "utils.hpp"

void im2col(float *input, float *output, size_t iW, size_t iH, size_t oH, size_t oW, size_t kH, size_t kW, size_t nChannel, size_t stride_h, size_t stride_w);
void direct_conv2d(const core::Tensor &input, const core::Tensor &kernel, core::Tensor &out, size_t stride);
void conv2d_neon(const float *input, const float *kernel, float *output, size_t stride,
                 const std::vector<int> outShape, const std::vector<int> kernelShape,
                 const std::vector<int> inStrides, const std::vector<int> kernelStrides,
                 const std::vector<int> outStrides);
void conv2d(const float *input, const float *kernel, float *output, size_t stride,
            const std::vector<int> outShape, const std::vector<int> kernelShape,
            const std::vector<int> inStrides, const std::vector<int> kernelStrides,
            const std::vector<int> outStrides);