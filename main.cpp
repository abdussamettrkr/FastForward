#include "binary_primitives.hpp"
#include "utils.hpp"
using namespace core;

int main()
{

    Tensor t = Tensor::ones({3, 2, 3});
    Tensor t2 = Tensor::ones({1});
    Tensor t3 = Tensor::ones({3, 2, 3});
    Add add;
    add.eval({t, t2}, t3);
    Tensor t4 = broadcastTo(t2, {4,3,2,3});
    
    
    std::cout << t3.data()[0] << "|" << t3.data()[1] << "|" << t3.data()[2] << "|" << t3.data()[3] << "|" << t3.data()[4] << "|" << t3.data()[5] << "|" << std::endl;
    std::cout << t3.getStrides()[0]  << "|"  << t3.getStrides()[1] << "|"<< t3.getStrides()[2] << "|" <<std::endl;
    std::cout << t4.getStrides()[0]  << "|"  << t4.getStrides()[1] << "|"<< t4.getStrides()[2] <<"|" << t4.getStrides()[3] << "|" <<std::endl;
}