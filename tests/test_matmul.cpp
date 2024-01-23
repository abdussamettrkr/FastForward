#include "tensor.hpp"

bool test_matmul()
{
    Tensor left = Tensor::ones({3, 4});
    Tensor right = Tensor::ones({4, 4});
    left.data()[3] = 2;
    left.data()[4] = 9;
    left.data()[6] = 2.5;
    left.data()[9] = 1.5;

    right.data()[0] = 7;
    right.data()[2] = 2.8;
    right.data()[5] = 1.9;
    right.data()[12] = 3.9;
    right.data()[14] = 10.71;

    float arr[] = {16.8, 5.9, 26.22, 5, 70.4, 14.4, 39.41, 13.5, 13.4, 5.85, 16.01, 4.5};
    Tensor expected({3, 4}, arr);
    Tensor result = left.matmul(right);

    return !(result == expected);
}

int main()
{
    return test_matmul();
}