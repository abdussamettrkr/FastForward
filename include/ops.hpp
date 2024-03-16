#pragma once
#include "tensor.hpp"
#include "utils.hpp"
#include "binary_primitives.hpp"
#include "unary_primitives.hpp"


core::Tensor binary_op(const core::Tensor& left, const core::Tensor& right, core::Primitive& op){
    if(!checkBroadcastable(left, right))
        throw std::logic_error("Tensors are not compatiable!");
    
    auto bshape = broadcastShapes(left.shape()->dims(), right.shape()->dims());
    auto bleft = broadcastTo(left, bshape);
    auto bright = broadcastTo(right, bshape);
    size_t out_size = 1;
    for (auto dim : bshape)
        out_size *= dim;
    
    auto out = core::Tensor(bshape, new float[out_size]);
    op.eval({bleft, bright}, out);
    return out;
}


namespace ops{
core::Tensor unary_op(const core::Tensor& in, core::Primitive& op){    
    auto out = core::Tensor(in.shape()->dims(), new float[in.size()]);
    op.eval({in}, out);
    return out;
}



core::Tensor add(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Add op;
    return binary_op(left, right, op);
}

core::Tensor substract(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Substract op;
    return binary_op(left, right, op);
}

core::Tensor divide(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Divide op;
    return binary_op(left, right, op);
}

core::Tensor multiply(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Divide op;
    return binary_op(left, right, op);
}

core::Tensor log(const core::Tensor& in){
    core::Log op;
    return unary_op(in, op);
}

core::Tensor sqrt(const core::Tensor& in){
    core::Sqrt op;
    return unary_op(in, op);
}
}