#include "tensor.hpp"

int main()
{

    Tensor t = Tensor::ones({3, 3});
    Tensor t2 = Tensor::ones({3, 3});
    t.data()[0] = 5;

    Tensor t3 = t.matmul(t2);
    std::cout << "ekmek" << std::endl;
    std::cout << t.data()[0] << "|" << t.data()[1] << std::endl;
    std::cout << *(t.shape()) << std::endl;
    std::cout << (*t.shape())[1] << std::endl;
}