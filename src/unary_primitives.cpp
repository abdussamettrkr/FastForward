
#include "unary_primitives.hpp"
#include "iterator.hpp"
#include <cmath>

namespace core{
void Log::eval(const std::vector<core::Tensor> inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    unary_array_iterator(inputs[0], out, [](float a) { return log(a); });
}

void Sqrt::eval(const std::vector<core::Tensor> inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    unary_array_iterator(inputs[0], out, [](float a) { return sqrt(a); });
}
}

