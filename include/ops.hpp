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

core::Tensor conv2d(const core::Tensor& input, const core::Tensor& kernel);
core::Tensor matmul(const core::Tensor& left, const core::Tensor& right);

core::Tensor max(const core::Tensor&input, const std::vector<int>& axes);
}