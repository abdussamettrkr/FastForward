#pragma once

#include <iostream>
#include "shape.hpp"

class Tensor
{
public:
    template <typename Iterable>
    static Tensor ones(Iterable &arraylike);
    static Tensor ones(std::initializer_list<int> arraylike);
    template <typename Iterable>
    static Tensor zeros(Iterable &arraylike);
    static Tensor zeros(std::initializer_list<int> arraylike);

    //static Tensor randn(unsigned int rows, unsigned int cols);

    float *data() const;

    Shape *shape();

    Tensor operator+(const Tensor &other);
    template <typename T>
    Tensor operator+(T value);
    Tensor operator-(const Tensor &other);
    template <typename T>
    Tensor operator-(T value);
    Tensor operator/(const Tensor &other);
    template <typename T>
    Tensor operator/(T value);
    Tensor operator*(const Tensor &other);
    template <typename T>
    Tensor operator*(T value);

private:
    float *m_data;
    Shape *m_shape;

    template <typename Iterable>
    Tensor(Iterable &arraylike, float *data);

    int size();
};
