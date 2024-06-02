#include "unary_primitives.hpp"
#include "utils.hpp"
#include "conv.hpp"
#include "ops.hpp"
#include <arm_neon.h>

namespace core
{

    void Convolution::eval(const std::vector<core::Tensor> &inputs, core::Tensor &out)
    {

        gemm_conv2d(inputs[0], inputs[1], out, stride);
    }
}

void gemm_conv2d(const core::Tensor &input, const core::Tensor &kernel, core::Tensor &out, size_t stride)
{
    const std::vector<int> kShape = kernel.shape();
    core::Tensor colImg = ops::im2col(input, kShape[1], kShape[2], 0, stride);
    core::Tensor kernelFlat = kernel.flatten(1, 3);
    kernelFlat = kernelFlat.transpose();
    auto result1 = ops::matmul(colImg, kernelFlat);

    auto outShape = out.shape();
    auto resultShape = result1.shape();
    memcpy(out.data(), result1.data(), result1.size() * sizeof(float));
}

void direct_conv2d(const core::Tensor &input, const core::Tensor &kernel, core::Tensor &out, size_t stride)
{

    const std::vector<int> outShape = out.shape();
    const std::vector<int> kShape = kernel.shape();

    const float *iData = input.data();
    const float *kData = kernel.data();
    float *outData = out.data();

    const std::vector<int> iStrides = input.strides();
    const std::vector<int> kStrides = kernel.strides();
    const std::vector<int> outStrides = out.strides();

    if (kShape[3] % 16 == 0)
    {
        conv2d_neon(iData, kData, outData, stride, outShape, kShape, iStrides, kStrides, outStrides);
    }
    else
    {
        conv2d(iData, kData, outData, stride, outShape, kShape, iStrides, kStrides, outStrides);
    }
}

void conv2d_neon(const float *input, const float *kernel, float *output, size_t stride,
                 const std::vector<int> outShape, const std::vector<int> kernelShape,
                 const std::vector<int> inStrides, const std::vector<int> kernelStrides,
                 const std::vector<int> outStrides)
{
    size_t N = outShape[0];
    size_t H = outShape[1];
    size_t W = outShape[2];
    size_t C = kernelShape[3];
    size_t O = kernelShape[0];
    size_t kH = kernelShape[1];
    size_t kW = kernelShape[2];

    for (size_t n = 0; n < N; n++)
    {
        for (size_t o = 0; o < O; o++)
        {
            for (size_t h = 0; h < H; h++)
            {
                for (size_t w = 0; w < W; w++)
                {
                    float32x4_t res1 = vdupq_n_f32(0.0f);
                    float32x4_t res2 = vdupq_n_f32(0.0f);
                    float32x4_t res3 = vdupq_n_f32(0.0f);
                    float32x4_t res4 = vdupq_n_f32(0.0f);
                    for (size_t kh = 0; kh < kH; kh++)
                    {
                        for (size_t kw = 0; kw < kW; kw++)
                        {
                            for (size_t c = 0; c < C; c += 16)
                            {
                                float32x4_t va1 = vld1q_f32(kernel + (o * kernelStrides[0] + kh * kernelStrides[1] + kw * kernelStrides[2] + c));
                                float32x4_t vb1 = vld1q_f32(input + (n * inStrides[0] + (h * stride + kh) * inStrides[1] + (w * stride + kw) * inStrides[2] + c));
                                float32x4_t va2 = vld1q_f32(kernel + (o * kernelStrides[0] + kh * kernelStrides[1] + kw * kernelStrides[2] + c + 4));
                                float32x4_t vb2 = vld1q_f32(input + (n * inStrides[0] + (h * stride + kh) * inStrides[1] + (w * stride + kw) * inStrides[2] + c + 4));
                                float32x4_t va3 = vld1q_f32(kernel + (o * kernelStrides[0] + kh * kernelStrides[1] + kw * kernelStrides[2] + c + 8));
                                float32x4_t vb3 = vld1q_f32(input + (n * inStrides[0] + (h * stride + kh) * inStrides[1] + (w * stride + kw) * inStrides[2] + c + 8));
                                float32x4_t va4 = vld1q_f32(kernel + (o * kernelStrides[0] + kh * kernelStrides[1] + kw * kernelStrides[2] + c + 12));
                                float32x4_t vb4 = vld1q_f32(input + (n * inStrides[0] + (h * stride + kh) * inStrides[1] + (w * stride + kw) * inStrides[2] + c + 12));
                                res1 = vfmaq_f32(res1, va1, vb1);
                                res2 = vfmaq_f32(res2, va2, vb2);
                                res3 = vfmaq_f32(res3, va3, vb3);
                                res4 = vfmaq_f32(res4, va4, vb4);
                            }
                        }
                    }
                    res1 = vaddq_f32(res1, res2);
                    res1 = vaddq_f32(res1, res3);
                    res1 = vaddq_f32(res1, res4);
                    float32x2_t sum_pairwise = vadd_f32(vget_low_f32(res1), vget_high_f32(res1));
                    output[n * outStrides[0] + h * outStrides[1] + w * outStrides[2] + o] += vget_lane_f32(vpadd_f32(sum_pairwise, sum_pairwise), 0);
                }
            }
        }
    }
}

void conv2d(const float *input, const float *kernel, float *output, size_t stride,
            const std::vector<int> outShape, const std::vector<int> kernelShape,
            const std::vector<int> inStrides, const std::vector<int> kernelStrides,
            const std::vector<int> outStrides)
{

    size_t N = outShape[0];
    size_t H = outShape[1];
    size_t W = outShape[2];
    size_t C = kernelShape[3];
    size_t O = kernelShape[0];
    size_t kH = kernelShape[1];
    size_t kW = kernelShape[2];
    for (size_t n = 0; n < N; n++)
    {
        for (size_t h = 0; h < H; h++)
        {
            for (size_t kh = 0; kh < kH; kh++)
            {
                for (size_t kw = 0; kw < kW; kw++)
                {
                    for (size_t c = 0; c < C; c++)
                    {
                        for (size_t w = 0; w < W; w++)
                        {
                            for (size_t o = 0; o < O; o++)
                            {
                                output[n * outStrides[0] + h * outStrides[1] + w * outStrides[2] + o] += kernel[o * kernelStrides[0] + kh * kernelStrides[1] + kw * kernelStrides[2] + c] * input[n * inStrides[0] + (h * stride + kh) * inStrides[1] + (w * stride + kw) * inStrides[2] + c];
                            }
                        }
                    }
                }
            }
        }
    }
}
