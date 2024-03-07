#include "tensor.hpp"
#include "iterator.hpp"
#include "utils.hpp"

namespace core{
template <typename Op>
void binary_op(const Tensor inp1, const Tensor inp2, Tensor &out, Op op){
    binary_array_iterator(inp1, inp2, out, op);
}
}