#pragma once
#include "tensor.hpp"

bool checkBroadcastable(const core::Tensor t1, const core::Tensor t2);
std::vector<int> broadcastShapes(const std::vector<int>& shape1, const std::vector<int>& shape2);
std::vector<int> squeezeShape(const std::vector<int> inputShape);
core::Tensor broadcastTo(const core::Tensor &t1, const std::vector<int> shape);
std::vector<int> calculateStride(const std::vector<int> shape);