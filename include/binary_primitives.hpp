#pragma once
#include "primitives.hpp"
#include "tensor.hpp"


namespace core{
class Add : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Substract : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Multiply : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Divide : public Primitive
{
    public:
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
};

class Matmul : public Primitive
{
    public:
        Matmul(bool _is_transposed): is_transposed(_is_transposed) {};
        Matmul(){
            is_transposed = false;
        };
        void eval(const std::vector<Tensor>& inputs, Tensor& out) override;
    private:
        bool is_transposed = false;
};
}
