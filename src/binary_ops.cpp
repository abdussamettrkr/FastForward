#include "tensor.hpp"
#include "utils.hpp"

namespace core{
template <typename Op>
void binary_op(const Tensor inp1, const Tensor inp2, Tensor &out, Op op){
    binary_array_iterator(inp1, inp2, out, op);
}


//This is disgusting
template <typename Op>
void binary_array_iterator(const Tensor inp1, const Tensor inp2, Tensor &out, Op op){
    if (!checkBroadcastable(inp1, inp2))
        throw std::logic_error("Tensors are not compatiable!");

    const float* inp1_data = inp1.data();
    const float* inp2_data = inp2.data();

    // Add parallel for
    for(size_t i = 0; i < inp1.size(); i++){
        out[i] = op(inp1_data[i], inp2_data[i]);
    }
}
}