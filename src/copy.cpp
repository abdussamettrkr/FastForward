#include "copy.hpp"

void copy(const float* source,float *target, size_t size, size_t s_offset, size_t t_offset, std::vector<int> shape, std::vector<int> strides){
    for (size_t i = 0; i < size; i++)
    {
        target[loc(i, shape, strides)+t_offset] = source[i+ s_offset];
    }
}