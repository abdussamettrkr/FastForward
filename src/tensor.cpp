#include "tensor.hpp"
#include "utils.hpp"
#include "ops.hpp"
#include "binary_primitives.hpp"


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
    std::vector<int> tensor1Shape = squeezeShape(this->shape()->dims());
    std::vector<int> tensor2Shape = squeezeShape(other.shape()->dims());

    if (tensor1Shape.size() < 2 || tensor2Shape.size() < 2)
        throw std::invalid_argument("Matmul currently support only 2 dim tensors");

    if (tensor1Shape[tensor1Shape.size() - 1] != tensor2Shape[tensor2Shape.size() - 2])
        throw std::invalid_argument("The ncols of the first tensor must be equal nrows! second tensor");

    if (!checkBroadcastable(*this, other))
        throw std::invalid_argument("Given tensors are not compatiable for mutmul operation!");

    size_t M = tensor1Shape.at(tensor1Shape.size() - 2);
    size_t N = tensor1Shape.back();
    size_t K = tensor2Shape.back();

    std::vector<int> resultShape;
    if (tensor1Shape.size() >= tensor2Shape.size())
        resultShape = tensor1Shape;
    else
        resultShape = tensor2Shape;

    resultShape.at(resultShape.size() - 2) = M;
    resultShape.back() = K;

    int broadCastedDims = 1;
    for (int i = 0; i < resultShape.size() - 2; i++)
        broadCastedDims *= resultShape[i];

    int t1ExtraDims, t2ExtraDims = 1;
    for (int i = 0; i < tensor1Shape.size() - 2; i++)
        t1ExtraDims *= tensor1Shape[i];
    for (int i = 0; i < tensor2Shape.size() - 2; i++)
        t2ExtraDims *= tensor2Shape[i];

    Tensor result = zeros(resultShape);

    for (int i = 0; i < broadCastedDims; i++)
    {
        float *t1Data = this->data() + ((i % t1ExtraDims) * M * N);
        float *t2Data = other.data() + ((i % t2ExtraDims) * M * K);
        float *resultData = result.data() + (i * M * K);

        for (int row = 0; row < resultShape[resultShape.size() - 2]; row++)
            for (int col = 0; col < resultShape.back(); col++)
                for (int inner = 0; inner < tensor1Shape.back(); inner++)
                    resultData[row * K + col] += t1Data[row * N + inner] * t2Data[K * inner + col];
    }
    return result;
}

std::vector<int> Tensor::getStrides() const{
    return this->strides;
}

}