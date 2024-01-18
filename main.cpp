// #include "tensor.hpp"
#include "conv2d.hpp"
int main()
{
    int size = 5;
    Tensor t = Tensor::ones({3, 3, size, size});
    // Tensor t2 = Tensor::ones({3, 3});
    // t.data()[0] = 5;

    // Tensor t3 = t.matmul(t2);
    // std::cout << "ekmek" << std::endl;
    // std::cout << t.data()[0] << "|" << t.data()[1] << std::endl;
    // std::cout << *(t.shape()) << std::endl;
    // std::cout << (*t.shape())[1] << std::endl;

    Conv2D conv(3, 1, 1); 
    Tensor convolved = conv(t);
    std::cout << t << std::endl;
    std::cout << convolved << std::endl;    

}