#include "tensor.hpp"

int main()
{

    Tensor t = Tensor::ones({3, 2, 3});
    Tensor t2 = Tensor::ones({3, 3});
    t.data()[0] = 5;
    t2.data()[2] = 0.25;

    Tensor t3 = t.matmul(t2);
    std::cout << "ekmek" << std::endl;
    std::cout << t.data()[0] << "|" << t.data()[1] << std::endl;
    std::cout << *(t.shape()) << std::endl;
    std::cout << *(t2.shape()) << std::endl;

    std::cout << t3.data()[0] << "|" << t3.data()[1] << "|" << t3.data()[2] << "|" << t3.data()[3] << "|" << t3.data()[4] << "|" << t3.data()[5] << "|" << std::endl;
    std::cout << *t3.shape() << std::endl;
}