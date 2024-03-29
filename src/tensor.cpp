#include "tensor.hpp"
#include "utils.hpp"
#include "ops.hpp"


namespace core{

Tensor Tensor::ones(std::vector<int> shape)
{
    int size = 1;
    for (auto &dim : shape)
        size *= dim;
    float *data = new float[size];
    for (int i = 0; i < size; i++)
    {
        *(data + i) = 1;
    }
    return Tensor(shape, data);
}


Tensor Tensor::zeros(std::vector<int> shape)
{
    int size = 1;
    for (auto &dim : shape)
        size *= dim;
    float *data = new float[size];
    for (int i = 0; i < size; i++)
    {
        *(data + i) = 0;
    }
    return Tensor(shape, data);
}

float *Tensor::data() const
{
    return storage->data;
}

float *Tensor::data()
{
    return storage->data;
}

int Tensor::size() const{ return storage->size; }
int Tensor::ndim() const{ return storage->ndim; }
std::vector<int> Tensor::shape() const { return storage->shape; }
std::vector<int> Tensor::strides() const{ return storage->strides; }
bool Tensor::is_contiguous() const { return storage->contiguous; }



Tensor Tensor::operator+(const Tensor &other)
{
    return ops::add(*this, other);
}

Tensor Tensor::operator-(const Tensor &other)
{
    return ops::substract(*this, other);
}

Tensor Tensor::operator/(const Tensor &other)
{
    return ops::divide(*this, other);
}

Tensor Tensor::operator*(const Tensor &other)
{
    return ops::multiply(*this, other);
}

bool Tensor::operator==(const Tensor &other){
    // TODO: instead of bool, return same shape bool Tensor
    if(storage->shape != other.storage->shape)
        throw("Tensor shapes must be equal!");
    
    for (size_t i = 0; i < storage->size; i++)
        if (std::abs(data()[i] - other.data()[i]) > EPSILON){
            std::cout <<"wrong one is:" <<i << std::endl;
            return false;
        }
            

    return true;
}

Tensor Tensor::max(std::vector<int> axes, bool keepdims){
    return ops::max(*this, axes, keepdims);
}

Tensor Tensor::sum(std::vector<int> axes, bool keepdims){
    return ops::sum(*this, axes, keepdims);
}

Tensor Tensor::prod(std::vector<int> axes, bool keepdims){
    return ops::prod(*this, axes, keepdims);
}

Tensor Tensor::min(std::vector<int> axes, bool keepdims){
    return ops::min(*this, axes, keepdims);
}


float& Tensor::operator[](int idx){
    return storage->data[idx];
}


Tensor::Tensor(const std::vector<int> shape, float *data)
{   
    std::vector<int> strides = calculateStride(shape);
    storage = new Storage(data, shape, strides);
}

Tensor::Tensor(const std::vector<int> shape)
{
    size_t size = 1;
    for (auto elem : shape)
    {
        size*=elem;
    }
    
    float * data = new float[size];
    storage = new Storage(data, shape, calculateStride(shape));
}


Tensor::Tensor(std::vector<int> shape, std::vector<int> strides, float *data){
    storage = new Storage(data, shape, strides);
}


Tensor::Tensor(const std::vector<Tensor>& _inputs, Primitive _op){
throw std::logic_error("Not implemented");
}


Tensor Tensor::matmul(const Tensor &other)
{
    return ops::matmul(*this, other);
}

}