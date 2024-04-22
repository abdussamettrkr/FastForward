#pragma once

#include <iostream>

#define EPSILON 1e-3

namespace core{
class Primitive;

class Tensor
{
public:
    Tensor(std::vector<int> shapes, std::vector<int> strides, float *data);
    Tensor(const std::vector<Tensor>& inputs, Primitive op);
    Tensor(const std::vector<int> arraylike, float *data);
    Tensor(const std::vector<int> shape);

    // Creation methods
    static Tensor ones(std::vector<int> shape);
    static Tensor zeros(std::vector<int> shape);
    // static Tensor randn(unsigned int rows, unsigned int cols);

    // Operators
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

    Tensor max(std::vector<int> axes = {}, bool keepdims = false);
    Tensor min(std::vector<int> axes = {}, bool keepdims = false);
    Tensor prod(std::vector<int> axes = {}, bool keepdims = false);
    Tensor sum(std::vector<int> axes = {}, bool keepdims = false);

    
    bool operator==(const Tensor &other);
    float& operator[](int index);

    Tensor matmul(const Tensor &other);
    Tensor flatten(size_t start_dim, size_t end_dim);

    float *data() const;
    float *data();
    int size() const;
    int ndim() const;
    std::vector<int> strides() const;
    std::vector<int> shape() const;
    bool is_contiguous() const;

private:
    class Storage{
        public:
            Storage(float *_data, std::vector<int> _shape, std::vector<int> _strides): 
                data(_data), shape(_shape), strides(_strides), ndim(_shape.size()) {
                    size=1;
                    for (size_t i : _shape)
                        size *= i;
                    contiguous = true;
                }

            float *data;
            size_t size;
            size_t ndim;
            bool contiguous;
            const std::vector<int> shape;
            const std::vector<int> strides;
    };
    Storage* storage;
};

// Template functions definition
template <typename T>
Tensor Tensor::operator+(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        storage->data[i] += value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator-(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        storage->data[i] -= value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator/(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        storage->data[i] /= value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator*(T value)
{
    int total_size = size();
    for (int i = 0; i < total_size; i++)
    {
        storage->data[i] *= value;
    }

    return *this;
}
}