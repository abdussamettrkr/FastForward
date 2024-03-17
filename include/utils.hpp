#pragma once
#include "tensor.hpp"

bool checkBroadcastable(const std::vector<int>& t1Shape, const std::vector<int>& t2Shape);
std::vector<int> broadcastShapes(const std::vector<int>& shape1, const std::vector<int>& shape2);
std::vector<int> squeezeShape(const std::vector<int> inputShape);
core::Tensor broadcastTo(const core::Tensor &t1, const std::vector<int> shape);
std::vector<int> calculateStride(const std::vector<int> shape);
size_t loc(size_t idx, const std::vector<int> &shapes, const std::vector<int> &strides);