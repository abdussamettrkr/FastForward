//#include "utils.hpp"
#include "ops.hpp"

int main()
{
    core::Tensor input = core::Tensor::ones({1, 5, 5, 4});
    core::Tensor kernel = core::Tensor::zeros({8, 3, 3, 4});
    kernel = kernel + 2;
    core::Tensor result = ops::conv2d(input, kernel, 2);

    std::cout << "<";
    for (auto item : result.shape())
        std::cout << item << ",";
    std::cout << ">" << std::endl;
    
    
    std::cout << result.data()[0] << "|" << result.data()[1] << "|" << result.data()[2] << "|" << result.data()[3] << "|" << result.data()[4] << "|" << result.data()[5] << "|" << std::endl;
    // std::cout << t3.getStrides()[0]  << "|"  << t3.getStrides()[1] << "|"<< t3.getStrides()[2] << "|" <<std::endl;
    std::cout << kernel.strides()[0]  << "|"  << kernel.strides()[1] << "|"<< kernel.strides()[2] <<"|" << kernel.strides()[3] << "|" <<std::endl;
}