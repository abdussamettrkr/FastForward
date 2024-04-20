#include "tensor.hpp"

using namespace core;

bool tensor_equal(const Tensor& t1, const Tensor& t2) {
    if (t1.shape() != t2.shape()) return false;
    for (size_t i = 0; i < t1.size(); ++i) {
        if (std::abs(t1.data()[i] - t2.data()[i]) > 1e-6) return false; // tolerance for floating point comparison
    }
    return true;
}

bool test_matmul()
{
    
    Tensor left1 = Tensor::ones({3, 4});
    Tensor right1 = Tensor::ones({4, 4});
    float expected1data[] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    Tensor expected1({3, 4}, expected1data);
    Tensor result1 = left1.matmul(right1);
    if (tensor_equal(result1, expected1) != true)
        return true;

    Tensor left2 = Tensor::zeros({3, 4});
    Tensor right2 = Tensor::ones({4, 4});
    float expected2data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Tensor expected2({3, 4}, expected2data);
    Tensor result2 = left2.matmul(right2);
    if (tensor_equal(result2, expected2) != true)
        return true;

    float left3data[] = {1, 2, 3, 4, 5, 6};
    Tensor left3({2, 3}, left3data);
    float right3data[] = {1, 2, 3, 4, 5, 6};
    Tensor right3({3, 2}, right3data);
    float expected3data[] = {22, 28, 49, 64};
    Tensor expected3({2, 2}, expected3data);
    Tensor result3 = left3.matmul(right3);
    if (tensor_equal(result3, expected3) != true)
        return true;
    return false;
}

int main()
{
    return test_matmul();
}