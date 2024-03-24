
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

    if (type == ReductionMethod::ContiguousAllReduce){
        reduce_contiguous_all(inputs[0], out, std::numeric_limits<float>::lowest(), [](float *a, float b) { *a = (*a > b) ? *a : b; });
    }
    else if(type == ReductionMethod::ContiguousReduce){
        reduce_contiguous(inputs[0], out, axes, std::numeric_limits<float>::lowest(), [](float *a, float b) { *a = (*a > b) ? *a : b; });
    }
}

void Min::eval(const std::vector<Tensor>& inputs, Tensor& out){

    if (type == ReductionMethod::ContiguousAllReduce){
        reduce_contiguous_all(inputs[0], out, std::numeric_limits<float>::max(), [](float *a, float b) { *a = (*a < b) ? *a : b; });
    }
    else if(type == ReductionMethod::ContiguousReduce){
        reduce_contiguous(inputs[0], out, axes, std::numeric_limits<float>::max(), [](float *a, float b) { *a = (*a < b) ? *a : b; });
    }
}

void Sum::eval(const std::vector<Tensor>& inputs, Tensor& out){

    if (type == ReductionMethod::ContiguousAllReduce){
        reduce_contiguous_all(inputs[0], out, 0, [](float *a, float b) { *a = *a + b; });
    }
    else if(type == ReductionMethod::ContiguousReduce){
        reduce_contiguous(inputs[0], out, axes, 0, [](float *a, float b) { *a =  *a + b; });
    }
}

void Prod::eval(const std::vector<Tensor>& inputs, Tensor& out){

    if (type == ReductionMethod::ContiguousAllReduce){
        reduce_contiguous_all(inputs[0], out, 1, [](float *a, float b) { *a = *a * b; });
    }
    else if(type == ReductionMethod::ContiguousReduce){
        reduce_contiguous(inputs[0], out, axes, 1, [](float *a, float b) { *a = *a * b; });
    }
}

}

