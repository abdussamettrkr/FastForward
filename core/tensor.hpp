#pragma once

#include <vector>
#include <iostream>


class Tensor{
    public:
        // ~Tensor() {
        //     free(m_data);
        // }
        static Tensor ones(unsigned int rows, unsigned int cols);
        static Tensor zeros(unsigned int rows, unsigned int cols);
        //static Tensor randn(unsigned int rows, unsigned int cols);
        

        float* data() const;

        //Shape* shape(); # Must be implemented 

        
        Tensor operator +(const Tensor &other);
        template <typename T>
        Tensor operator +(T value);
        Tensor operator -(const Tensor &other);
        template <typename T>
        Tensor operator -(T value);
        Tensor operator /(const Tensor &other);
        template <typename T>
        Tensor operator /(T value);
        Tensor operator *(const Tensor &other);
        template <typename T>
        Tensor operator *(T value);


    private:
        float* m_data;  
        std::vector <unsigned int> m_shape;
        unsigned int m_rows;
        unsigned int m_cols;
        unsigned int n_dim;

        Tensor(unsigned int rows, unsigned int cols, float* data);

        int size();
};