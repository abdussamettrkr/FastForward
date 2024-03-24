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

class Reduce : public Primitive
{
    public:
        Reduce(ReductionMethod _type, std::vector<int> _axes): type(_type), axes(_axes) {}
    protected:
        ReductionMethod type;
        std::vector<int> axes;
};

class Max : public Reduce{
    public:
        Max(ReductionMethod _type, std::vector<int> _axes) : Reduce(_type, _axes) {}
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Min : public Reduce{
    public:
        Min(ReductionMethod _type, std::vector<int> _axes) : Reduce(_type, _axes) {}
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Sum : public Reduce{
    public:
        Sum(ReductionMethod _type, std::vector<int> _axes) : Reduce(_type, _axes) {}
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Prod : public Reduce{
    public:
        Prod(ReductionMethod _type, std::vector<int> _axes) : Reduce(_type, _axes) {}
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};
}
