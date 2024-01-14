#pragma once

#include <iostream>
#include "shape.hpp"

class Tensor
{
public:
    Shape *shape() const;

    template <typename Iterable>
    static Tensor ones(Iterable &arraylike);
    static Tensor ones(std::initializer_list<int> arraylike);
    template <typename Iterable>
    static Tensor zeros(Iterable &arraylike);
    static Tensor zeros(std::initializer_list<int> arraylike);

    //static Tensor randn(unsigned int rows, unsigned int cols);

    template <typename T>
    Tensor operator+(T value);
    Tensor operator+(const Tensor &other);

    template <typename T>
    Tensor operator-(T value);
    Tensor operator-(const Tensor &other);

    template <typename T>
    Tensor operator/(T value);
    Tensor operator/(const Tensor &other);

    template <typename T>
    Tensor operator*(T value);
    Tensor operator*(const Tensor &other);

    Tensor matmul(const Tensor &other);

    float *data() const;

private:
    float *m_data;
    Shape *m_shape;

    template <typename Iterable>
    Tensor(Iterable &arraylike, float *data);

    int size();
};

// Template functions definition
template <typename T>
Tensor Tensor::operator+(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] += value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator-(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] -= value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator/(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] /= value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator*(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        m_data[i] *= value;
    }

    return *this;
}
