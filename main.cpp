#include "tensor.hpp"

int main()
{

    Tensor t = Tensor::ones({300, 400, 1000});
    Tensor t2 = Tensor::ones({300, 400, 1000});

    t = t + t2;

    t = t + t;
    t = t * 4.f;

    t = t / 3.f;
    std::cout << "ekmek" << std::endl;
    std::cout << t.data()[0] << "|" << t.data()[1] << std::endl;
    std::cout << *(t.shape()) << std::endl;
}