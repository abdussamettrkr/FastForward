#include "tensor.hpp"
#include "utils.hpp"
#include "ops.hpp"


namespace core{
template <typename Iterable>
Tensor Tensor::ones(Iterable &arraylike)
{
    int n_ones = 1;
    for (auto &dim : arraylike)
        n_ones *= dim;
    float *data = new float[n_ones];
    for (int i = 0; i < n_ones; i++)
    {
        *(data + i) = 1;
    }
    return Tensor(arraylike, data);
}

Tensor Tensor::ones(std::initializer_list<int> arraylike)
{
    return Tensor::ones<decltype(arraylike)>(arraylike);
}

Tensor Tensor::zeros(std::initializer_list<int> arraylike)
{
    return Tensor::zeros<decltype(arraylike)>(arraylike);
}

float *Tensor::data() const
{
    return m_data;
}

float *Tensor::data()
{
    return m_data;
}

int Tensor::size() const{ return m_shape->size(); };

Shape *Tensor::shape() const
{
    return m_shape;
}

// We will perform operations inplace!
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
    if(this->shape()->dims() != other.shape()->dims())
        throw("Tensor shapes must be equal!");
    
    for (size_t i = 0; i < this->shape()->size(); i++)
        if (std::abs(data()[i] - other.data()[i]) > EPSILON){
            std::cout <<"wrong one iss:" <<i << std::endl;
            return false;
        }
            

    return true;
}

Tensor Tensor::log(){
return ops::log(*this);
}

Tensor Tensor::sqrt(){
return ops::sqrt(*this);
}

float& Tensor::operator[](int idx){
    return m_data[idx];
}

template <typename Iterable>
Tensor::Tensor(Iterable &arraylike, float *data)
{
    m_data = data;
    this->m_shape = new Shape(arraylike);
    this->strides = calculateStride(m_shape->dims());
}

Tensor::Tensor(const std::vector<int> arraylike, float *data)
{
    m_data = data;
    this->m_shape = new Shape(arraylike);
    this->strides = calculateStride(m_shape->dims());
}

Tensor::Tensor(const std::vector<int> shape)
{
    size_t size = 1;
    for (auto elem : shape)
    {
        size*=elem;
    }
    
    m_data = new float[size];
    this->m_shape = new Shape(shape);
    this->strides = calculateStride(m_shape->dims());
}

Tensor::Tensor(std::initializer_list<int> arraylike, float *data){
    m_data = data;
    this->m_shape = new Shape(arraylike);
    this->strides = calculateStride(m_shape->dims());
}

Tensor::Tensor(std::vector<int> shapes, std::vector<int> strides, float *data){
    m_data = data;
    this->m_shape = new Shape(shapes);
    this->strides = strides;
}


Tensor::Tensor(const std::vector<Tensor>& _inputs, Primitive _op){
throw std::logic_error("Not implemented");
}

template <typename Iterable>
Tensor Tensor::zeros(Iterable &arraylike)
{
    int n_zeros = 1;
    for (auto &dim : arraylike)
        n_zeros *= dim;
    float *data = new float[n_zeros];
    for (int i = 0; i < n_zeros; i++)
    {
        *(data + i) = 0;
    }
    return Tensor(arraylike, data);
}

Tensor Tensor::matmul(const Tensor &other)
{
    return ops::matmul(*this, other);
}

std::vector<int> Tensor::getStrides() const{
    return this->strides;
}

}