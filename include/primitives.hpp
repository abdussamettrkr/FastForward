#pragma once
#include "tensor.hpp"
#include <functional>

namespace core{
class Primitive
{
    public:
        virtual void eval(const std::vector<Tensor> inputs, Tensor& out){
            throw std::logic_error("Primitive eval not implemented!");
        };
};
}

