#include "utils.hpp"

void gemm_conv2d(const core::Tensor &input, const core::Tensor &kernel, core::Tensor &out, size_t stride);
void direct_conv2d(const core::Tensor &input, const core::Tensor &kernel, core::Tensor &out, size_t stride);
void conv2d_neon(const float *input, const float *kernel, float *output, size_t stride,
                 const std::vector<int> outShape, const std::vector<int> kernelShape,
                 const std::vector<int> inStrides, const std::vector<int> kernelStrides,
                 const std::vector<int> outStrides);
void conv2d(const float *input, const float *kernel, float *output, size_t stride,
            const std::vector<int> outShape, const std::vector<int> kernelShape,
            const std::vector<int> inStrides, const std::vector<int> kernelStrides,
            const std::vector<int> outStrides);