#include "tensor.hpp"
#include "utils.hpp"


template <typename Op>
void reduce_contiguous_all(const core::Tensor& input, core::Tensor& out, float init_val, Op op){

    const float * input_data = input.data();
    float *output_data = out.data();
    *output_data = init_val;
    for (size_t i = 0; i < input.size(); i++)
    {
        op(output_data, input_data[i]);
    }
}


template <typename Op>
void reduce_contiguous_dim(const float *input_data, float *output_data, const std::vector<int> reduction_size, const std::vector<int> reduction_strides, size_t offset, size_t dim, Op op){
    
    if( reduction_size.size()-1 == dim){
        for (size_t i = 0; i < reduction_size.back(); i++)
        {
            op(output_data, input_data[offset + i * reduction_strides.back()]);
        }
    }
    else{
        for (size_t i = 0; i < reduction_size[dim]; i++)
        {
            reduce_contiguous_dim(input_data, output_data, reduction_size, reduction_strides, offset + i * reduction_strides[dim], dim+1, op);
        }
    }

}


template <typename Op>
void reduce_contiguous(const core::Tensor& input, core::Tensor& out, std::vector<int> axes, float init_val, Op op){
    const std::vector<int>& in_shapes = input.shape();
    const std::vector<int>& in_strides = input.strides();
    std::vector<int> reduce_size = {in_shapes[axes[0]]};
    std::vector<int> reduce_strides = {in_strides[axes[0]]};
    float *output_data = out.data();

    for (size_t i = 1; i < axes.size(); i++)
    {
        if(axes[i] -1 == axes[i-1]){
            reduce_size.back() *= in_shapes[axes[i]];
            reduce_strides.back() = in_strides[axes[i]];
        }
        else{
            reduce_size.push_back(in_shapes[axes[i]]);
            reduce_strides.push_back(in_strides[axes[i]]);
        }
    }
    
    for (size_t i = 0; i < out.size(); i++, output_data++)
    {
        *output_data = init_val;
        reduce_contiguous_dim(input.data(), output_data, reduce_size, reduce_strides, 0, 0, op);   
    }
    
}
