#include "tensor.hpp"
#include <iostream>


int main(){
    Tensor t = Tensor::ones(300, 400);
    Tensor t2 = Tensor::ones(300, 400);

    t = t + t2;
    t = t+t;
    t = t * 4.f;

    t = t / 3.f;


    std::cout << t.data()[0] << "|" << t.data()[1] << std::endl;

}