#pragma once

#include "tensor.hpp"
template <typename Iterable>
Tensor::Tensor(Iterable& arraylike, float* data){
    m_data = data;
    this->m_shape = new Shape(arraylike);
}

template <typename Iterable>
Tensor Tensor::ones(Iterable& arraylike){
    int n_ones = 1;
    for(auto &dim : arraylike)
        n_ones *= dim;
    float* data = new float[n_ones];
    for(int i = 0; i < n_ones; i ++){
        *(data+i) = 1;
    }
    return Tensor(arraylike, data);
}

template <typename Iterable>
Tensor Tensor::zeros(Iterable& arraylike){
    int n_zeros = 1;
    for(auto &dim : arraylike)
        n_zeros *= dim;
    float* data = new float[n_zeros];
    for(int i = 0; i < n_zeros; i ++){
        *(data+i) = 0;
    }
    return Tensor(arraylike, data);
}

template <typename T>
Tensor Tensor::operator+(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){        
        m_data[i] += value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator-(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] -= value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator*(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] *= value;
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator/(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] /= value;
    }

    return *this;
}