#pragma once
#include "tensor.hpp"
#include "utils.hpp"
#include "binary_primitives.hpp"
#include "unary_primitives.hpp"



namespace ops{
core::Tensor unary_op(const core::Tensor& in, core::Primitive& op);
core::Tensor binary_op(const core::Tensor& left, const core::Tensor& right, core::Primitive& op);

core::Tensor add(const core::Tensor& left, const core::Tensor& right);
core::Tensor substract(const core::Tensor& left, const core::Tensor& right);
core::Tensor multiply(const core::Tensor& left, const core::Tensor& right);
core::Tensor divide(const core::Tensor& left, const core::Tensor& right);

core::Tensor log(const core::Tensor& in);
core::Tensor sqrt(const core::Tensor& in);
core::Tensor exp(const core::Tensor& in);
core::Tensor relu(const core::Tensor& in);


core::Tensor conv2d(const core::Tensor& input, const core::Tensor& kernel, size_t stride=1);
core::Tensor maxpool2d(const core::Tensor& input, size_t kernel_size, size_t stride);
core::Tensor matmul(const core::Tensor& left, const core::Tensor& right);
core::Tensor pad(const core::Tensor& input, std::vector<int> pad_width);
core::Tensor softmax(const core::Tensor& input, int axis = -1);

core::Tensor max(const core::Tensor&input, const std::vector<int>& axes, bool keepdims);
core::Tensor min(const core::Tensor&input, const std::vector<int>& axes, bool keepdims);
core::Tensor prod(const core::Tensor&input, const std::vector<int>& axes, bool keepdims);
core::Tensor sum(const core::Tensor&input, const std::vector<int>& axes, bool keepdims);
}