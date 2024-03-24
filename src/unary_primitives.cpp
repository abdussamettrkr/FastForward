
#include "unary_primitives.hpp"
#include "iterator.hpp"
#include "reduce.hpp"
#include <cmath>
#include <limits>

namespace core{
void Log::eval(const std::vector<core::Tensor>& inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    unary_array_iterator(inputs[0], out, [](float a) { return log(a); });
}

void Sqrt::eval(const std::vector<core::Tensor>& inputs, core::Tensor& out){
    // Instead of float use scalar_dtype 
    unary_array_iterator(inputs[0], out, [](float a) { return sqrt(a); });
}


void Max::eval(const std::vector<Tensor>& inputs, Tensor& out){

    if (type == ReductionType::ContiguousAllReduce){
        reduce_contiguous_all(inputs[0], out, std::numeric_limits<float>::lowest(), [](float *a, float b) { *a = (*a > b) ? *a : b; });
    }
    else if(type == ReductionType::ContiguousReduce){
        reduce_contiguous(inputs[0], out, axes, std::numeric_limits<float>::lowest(), [](float *a, float b) { *a = (*a > b) ? *a : b; });
    }
}
}

