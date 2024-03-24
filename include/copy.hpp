
#include <vector>
#include "utils.hpp"

void copy(const float* source,float *target, size_t size, size_t s_offset, size_t t_offset, std::vector<int> shape, std::vector<int> strides);