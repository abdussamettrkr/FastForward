
#include "binary_primitives.hpp"
#include "binary_ops.cpp"

namespace core{
void Add::eval(const std::vector<core::Tensor> inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    binary_op(inputs[0], inputs[1], out, [](float a, float b) { return a + b; });
}

void Substract::eval(const std::vector<core::Tensor> inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    binary_op(inputs[0], inputs[1], out, [](float a, float b) { return a + b; });
}

void Divide::eval(const std::vector<core::Tensor> inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    binary_op(inputs[0], inputs[1], out, [](float a, float b) { return a + b; });
}

void Multiply::eval(const std::vector<core::Tensor> inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    binary_op(inputs[0], inputs[1], out, [](float a, float b) { return a + b; });
}
}

