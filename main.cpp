//#include "utils.hpp"
#include "ops.hpp"

int main()
{
    core::Tensor input = core::Tensor::ones({1, 3, 3, 4});
    core::Tensor kernel = core::Tensor::ones({5, 3, 3, 4});
    core::Tensor result = ops::conv2d(input, kernel);

    std::cout << "<";
    for (auto item : result.shape()->dims())
        std::cout << item << ",";
    std::cout << ">" << std::endl;
    
    
    std::cout << result.data()[0] << "|" << result.data()[1] << "|" << result.data()[2] << "|" << result.data()[3] << "|" << result.data()[4] << "|" << result.data()[5] << "|" << std::endl;
    // std::cout << t3.getStrides()[0]  << "|"  << t3.getStrides()[1] << "|"<< t3.getStrides()[2] << "|" <<std::endl;
    std::cout << kernel.getStrides()[0]  << "|"  << kernel.getStrides()[1] << "|"<< kernel.getStrides()[2] <<"|" << kernel.getStrides()[3] << "|" <<std::endl;
}