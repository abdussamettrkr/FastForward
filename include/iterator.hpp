#pragma once
#include "tensor.hpp"
#include "utils.hpp"

template <typename Op>
void binary_array_iterator(const core::Tensor& left, const core::Tensor& right, core::Tensor &out, Op op){
    const float* left_data = left.data();
    const float* right_data = right.data();

    // TODO: Add parallel for
    for(size_t i = 0; i < out.size(); i++){
        // TODO: Use loc only when Tensor is not contiguous
        size_t left_idx = loc(i, left.shape(), left.strides());
        size_t right_idx = loc(i, right.shape(), right.strides());
        out[i] = op(left_data[left_idx], right_data[right_idx]);
    }
}


template <typename Op>
void unary_array_iterator(const core::Tensor& left, core::Tensor &out, Op op){
    const float* left_data = left.data();

    // TODO: Add parallel for
    for(size_t i = 0; i < out.size(); i++){
        out[i] = op(left_data[i]);
    }
}
