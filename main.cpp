#include "binary_primitives.hpp"

using namespace core;

int main()
{

    Tensor t = Tensor::ones({3, 2, 3});
    Tensor t2 = Tensor::ones({3, 2, 3});
    Tensor t3 = Tensor::ones({3, 2, 3});
    Add add;
    add.eval({t, t2}, t3);
    
    std::cout << t3.data()[0] << "|" << t3.data()[1] << "|" << t3.data()[2] << "|" << t3.data()[3] << "|" << t3.data()[4] << "|" << t3.data()[5] << "|" << std::endl;
    std::cout << t3.getStrides()[0]  << "|"  << t3.getStrides()[1] << "|"<< t3.getStrides()[2] << "|" <<std::endl;
}