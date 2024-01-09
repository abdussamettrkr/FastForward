#include "tensor.hpp"

Tensor Tensor::ones(std::initializer_list<int> arraylike){
    return Tensor::ones<decltype(arraylike)>(arraylike);
}

Tensor Tensor::zeros(std::initializer_list<int> arraylike){
    return Tensor::ones<decltype(arraylike)>(arraylike);
}

float* Tensor::data() const{
    return m_data;
}

int Tensor::size() {return m_shape->size();};


Shape* Tensor::shape() {return m_shape;}

// We will perform operations inplace!
Tensor Tensor::operator+(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] += other.data()[i];
    }

    return *this;
}

Tensor Tensor::operator-(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] -= other.data()[i];
    }

    return *this;
}

Tensor Tensor::operator*(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] *= other.data()[i];
    }

    return *this;
}

Tensor Tensor::operator/(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] /= other.data()[i];
    }

    return *this;
}
