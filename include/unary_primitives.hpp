#pragma once
#include "primitives.hpp"
#include "tensor.hpp"


namespace core{
class Log : public Primitive
{
    public:
        void eval(const std::vector<Tensor> inputs, Tensor& out) override;
};

class Sqrt : public Primitive
{
    public:
        void eval(const std::vector<Tensor> inputs, Tensor& out) override;
};

}
