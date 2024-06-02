#pragma once

#include <iostream>

#define EPSILON 1e-3

namespace core
{
    class Primitive;

    class Tensor
    {
    public:
        Tensor(std::vector<int> shapes, std::vector<int> strides, float *data);
        Tensor(const std::vector<Tensor> &inputs, Primitive op);
        Tensor(const std::vector<int> arraylike, float *data);
        Tensor(const std::vector<int> shape);
        Tensor();

        Tensor(Tensor &&other) noexcept : storage(other.storage)
        {
            other.storage = nullptr;
        }

        // // Move assignment operator
        // Tensor& operator=(Tensor&& other) noexcept {
        //     if (this != &other) {
        //         delete storage;
        //         storage = other.storage;
        //         other.storage = nullptr;
        //     }
        //     return *this;
        // }

        Tensor(const Tensor &other)
        {
            if (other.storage)
            {
                storage = new Storage(*other.storage);
            }
            else
            {
                storage = nullptr;
            }
        }

        Tensor &operator=(const Tensor &other)
        {
            if (this != &other)
            {
                if (storage){
                    delete storage;
                }
                
                if (other.storage)
                {
                    storage = new Storage(*other.storage);
                }
                else
                {
                    storage = nullptr;
                }
            }
            return *this;
        }

        // Creation methods
        static Tensor ones(std::vector<int> shape);
        static Tensor zeros(std::vector<int> shape);
        // static Tensor randn(unsigned int rows, unsigned int cols);

        // Operators
        template <
            typename T,
            typename = typename std::enable_if<std::is_same<T, int>::value || std::is_same<T, float>::value, Tensor>::type>
        Tensor operator+(const T value);
        Tensor operator+(const Tensor &other);

        template <
            typename T,
            typename = typename std::enable_if<std::is_same<T, int>::value || std::is_same<T, float>::value, Tensor>::type>
        Tensor operator-(const T value);
        Tensor operator-(const Tensor &other);

        template <typename T>
        Tensor operator/(const T value);
        Tensor operator/(const Tensor &other);

        template <typename T>
        Tensor operator*(const T value);
        Tensor operator*(const Tensor &other);

        Tensor max(std::vector<int> axes = {}, bool keepdims = false);
        Tensor min(std::vector<int> axes = {}, bool keepdims = false);
        Tensor prod(std::vector<int> axes = {}, bool keepdims = false);
        Tensor sum(std::vector<int> axes = {}, bool keepdims = false);

        Tensor transpose() const;

        bool operator==(const Tensor &other);
        float &operator[](int index);

        Tensor matmul(const Tensor &other);
        Tensor flatten(size_t start_dim, size_t end_dim) const;

        float *data() const;
        float *data();
        int size() const;
        int ndim() const;
        std::vector<int> strides() const;
        std::vector<int> shape() const;
        bool is_contiguous() const;

    private:
        class Storage
        {
        public:
            Storage(float *_data, std::vector<int> _shape, std::vector<int> _strides) : data(_data), shape(_shape), strides(_strides), ndim(_shape.size())
            {
                size = 1;
                for (size_t i : _shape)
                    size *= i;
                contiguous = true;
            }

            float *data;
            size_t size;
            size_t ndim;
            bool contiguous;
            std::vector<int> shape;
            std::vector<int> strides;
        };
        Storage *storage = nullptr;
        friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
        {
            auto shape = tensor.shape();
            float *data = tensor.data();
    
            os << "[";
            for (int i = 0; i < shape.at(shape.size() - 2); ++i)
            {
                if (shape.at(shape.size() - 2) > 6 && i == 3)
                {
                    os << "..., " << std::endl;
                    i = shape.at(shape.size() - 2) - 4;
                    continue;
                }

                os << "[";
                for (int j = 0; j < shape.back(); ++j)
                {
                    if (shape.back() > 6 && j == 3)
                    {
                        os << "..., ";
                        j = shape.back() - 4;
                        continue;
                    }

                    os << data[j + shape.back() * i];
                    if (j < shape.back() - 1)
                    {
                        os << ", ";
                    }
                }
                os << "]";
                if (i < shape.at(shape.size() - 2) - 1)
                {
                    os << std::endl;
                }
            }
            os << "]";
            return os;
        }
    };

    // Template functions definition
    template < typename T, typename>
    Tensor Tensor::operator+(const T value)
    {
        Tensor created(*this);
        int total_size = size();
        for (int i = 0; i < total_size; i++)
        {
            created.storage->data[i] += value;
        }

        return *this;
    }

    template <
        typename T, typename>
    Tensor Tensor::operator-(const T value)
    {
        Tensor created(*this);
        int total_size = size();
        for (int i = 0; i < total_size; i++)
        {
            created.storage->data[i] -= value;
        }

        return created;
    }

    template <typename T>
    Tensor Tensor::operator/(const T value)
    {
        Tensor created(*this);
        int total_size = size();
        for (int i = 0; i < total_size; i++)
        {
            created.storage->data[i] /= value;
        }

        return *this;
    }

    template <typename T>
    Tensor Tensor::operator*(const T value)
    {
        Tensor created(*this);
        int total_size = size();
        for (int i = 0; i < total_size; i++)
        {
            created.storage->data[i] *= value;
        }

        return *this;
    }
}