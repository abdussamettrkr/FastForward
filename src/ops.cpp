#include "ops.hpp"


namespace ops{
core::Tensor unary_op(const core::Tensor& in, core::Primitive& op){    
    auto out = core::Tensor(in.shape(), new float[in.size()]);
    op.eval({in}, out);
    return out;
}


core::Tensor binary_op(const core::Tensor& left, const core::Tensor& right, core::Primitive& op){
    if(!checkBroadcastable(left.shape(), right.shape()))
        throw std::logic_error("Tensors are not compatiable!");
    
    auto bshape = broadcastShapes(left.shape(), right.shape());
    auto bleft = broadcastTo(left, bshape);
    auto bright = broadcastTo(right, bshape);
    size_t out_size = 1;
    for (auto dim : bshape)
        out_size *= dim;
    
    auto out = core::Tensor(bshape, new float[out_size]);
    op.eval({bleft, bright}, out);
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

core::Tensor matmul(const core::Tensor& left, const core::Tensor& right){
    const std::vector<int>& leftShape = left.shape();
    const std::vector<int>& rightShape = right.shape();
    std::vector<int> leftBaseShape(leftShape.begin(), leftShape.end()-2);
    std::vector<int> rightBaseShape(rightShape.begin(), rightShape.end()-2);

    
    auto outShape = broadcastShapes(leftBaseShape, rightBaseShape);
    outShape.push_back(left.shape()[left.ndim()-2]);
    outShape.push_back(right.shape()[right.ndim()-1]);

    auto out = core::Tensor(outShape);

    core::Matmul op;
    op.eval({left, right}, out);   
    return out;
}

core::Tensor conv2d(const core::Tensor& input, const core::Tensor& kernel){
    // TODO: Check shapes are compatiable
    const std::vector<int>& inputShape = input.shape();
    const std::vector<int>& kernelShape = kernel.shape();

    std::vector<int> outShape(inputShape.begin(), inputShape.end()-1);
    outShape.insert(outShape.end(), kernelShape[0]);

    auto out = core::Tensor(outShape);
    core::Convolution op;
    op.eval({input, kernel}, out);
    return out;
}
}