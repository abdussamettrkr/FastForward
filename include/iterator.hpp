#include "tensor.hpp"
#include "utils.hpp"

template <typename Op>
void binary_array_iterator(const core::Tensor left, const core::Tensor right, core::Tensor &out, Op op){
    if (!checkBroadcastable(left, right))
        throw std::logic_error("Tensors are not compatiable!");

    auto bshape = broadcastShapes(left.shape()->dims(), right.shape()->dims());
    auto bleft = broadcastTo(left, bshape);
    auto bright = broadcastTo(right, bshape);

    const float* bleft_data = bleft.data();
    const float* bright_data = bright.data();

    // Add parallel for
    for(size_t i = 0; i < out.size(); i++){
        size_t bleft_idx = loc(i, bleft.shape()->dims(), bleft.getStrides()) / 8;
        size_t bright_idx = loc(i, bright.shape()->dims(), bright.getStrides()) / 8;
        out[i] = op(bleft_data[bleft_idx], bright_data[bright_idx]);
    }
}

