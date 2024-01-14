#include "tensor.hpp"

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
    int t1_h = (*this->shape())[0];
    int t1_w = (*this->shape())[1];
    int t2_h = (*other.shape())[0];
    int t2_w = (*other.shape())[1];

    if (this->shape()->ndims() != 2 || other.shape()->ndims() != 2)
        throw std::invalid_argument("Matmul currently support only 2 dim tensors");

    if (t1_w != t2_h)
        throw std::invalid_argument("The ncols of the first tensor must be equal nrows! second tensor");

    Tensor result = zeros({t1_h, t2_w});

    for (int row = 0; row < t1_h; row++)
    {
        for (int col = 0; col < t2_w; col++)
        {
            for (int inner = 0; inner < t1_w; inner++)
            {
                result.data()[row * t1_h + col] += this->data()[row * t1_w + inner] * other.data()[t2_w * inner + col];
            }
        }
    }
    return result;
}