#pragma once
#include "primitives.hpp"
#include "tensor.hpp"
#include "utils.hpp"


namespace core{
class Log : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Sqrt : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Convolution : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Max : public Primitive
{
    public:
        Max(ReductionType _type, std::vector<int> _axes): type(_type), axes(_axes) {}
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
    private:
        ReductionType type;
        std::vector<int> axes;
};
}
