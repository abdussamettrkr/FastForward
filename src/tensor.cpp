#include "tensor.hpp"
#include "utils.cpp"

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

int Tensor::size() { return m_shape->size(); };

Shape *Tensor::shape() const
{
    return m_shape;
}

// We will perform operations inplace!
Tensor Tensor::operator+(const Tensor &other)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] += other.data()[i];
    }

    return *this;
}

Tensor Tensor::operator-(const Tensor &other)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] -= other.data()[i];
    }

    return *this;
}

Tensor Tensor::operator/(const Tensor &other)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] /= other.data()[i];
    }

    return *this;
}

Tensor Tensor::operator*(const Tensor &other)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] *= other.data()[i];
    }

    return *this;
}

template <typename Iterable>
Tensor::Tensor(Iterable &arraylike, float *data)
{
    m_data = data;
    this->m_shape = new Shape(arraylike);
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
    std::vector<int> tensor1Shape = this->shape()->dims();
    std::vector<int> tensor2Shape = other.shape()->dims();

    if (tensor1Shape.size() < 2 || tensor2Shape.size() < 2)
        throw std::invalid_argument("Matmul currently support only 2 dim tensors");

    if (tensor1Shape[tensor1Shape.size() - 1] != tensor2Shape[tensor2Shape.size() - 2])
        throw std::invalid_argument("The ncols of the first tensor must be equal nrows! second tensor");

    if (!checkBroadcastable(*this, other))
        throw std::invalid_argument("Given tensors are not compatiable for mutmul operation!");

    std::vector<Tensor> broadCastedTensors = broadCast(*this, other);
    std::vector<int> resultShape;
    std::vector<int> broadCastedShape = (*broadCastedTensors[0].shape()).dims();
    int broadCastedDims = 1;
    for (int i = 0; i < broadCastedTensors.size() - 2; i++)
    {
        resultShape.push_back(broadCastedShape[i]);
    }

    resultShape.push_back(tensor1Shape[tensor1Shape.size() - 2]);
    resultShape.push_back(tensor2Shape.back());

    Tensor result = zeros(broadCastedShape);

    for (int i = 0; i < resultShape.size() - 2; i++)
    {
        broadCastedDims *= resultShape[i];
    }

    for (int i = 0; i < broadCastedDims; i++)
    {
        float *resultData = result.data() + (i * resultShape[resultShape.size() - 2] * resultShape.back());
        float *t1Data = this->data();
        float *t2Data = other.data();

        for (int row = 0; row < resultShape[resultShape.size() - 2]; row++)
        {
            for (int col = 0; col < resultShape.back(); col++)
            {
                for (int inner = 0; inner < tensor1Shape.back(); inner++)
                {
                    resultData[row * resultShape.back() + col] += t1Data[row * tensor1Shape.back() + inner] * t2Data[tensor2Shape.back() * inner + col];
                }
            }
        }
    }
    return result;
}