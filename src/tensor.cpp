#include "tensor.hpp"
#include "utils.cpp"

Tensor::Tensor(){
    std::vector<int> emptyShape(0);
    this->m_shape = new Shape(emptyShape);
}

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
    float outData[total_size];
    for (int i = 0; i < total_size; i++)
    {
        outData[i] = m_data[i] * other.data()[i];
    }
    std::vector<int> outShape = this->shape()->dims();
    return Tensor(outShape, outData);
}

template <typename Iterable>
Tensor::Tensor(Iterable &arraylike, float *data)
{
    m_data = data;
    this->m_shape = new Shape(arraylike);
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

Tensor Tensor::getKernel(float* buffer, float* fromMat, int kernel_size, int i, int j){
    //buffer should be of size kernel_size x kernel_size
    std::vector<int> dims = shape()->dims();
    size_t rows = dims[dims.size() - 2];
    size_t cols = dims[dims.size() - 1];
    float* center = fromMat + i * cols + j;
    float* head = center - kernel_size / 2 - (kernel_size / 2) * cols;
    for (size_t r = 0; r < kernel_size; r++){
        for (size_t c = 0; c < kernel_size; c++){
            buffer[r * kernel_size + c] = *(head + r * cols + c);
        }
        
    }
    std::vector<int> sliceShape = {kernel_size, kernel_size};
    return Tensor(sliceShape, buffer);
}

float Tensor::sum(){
    float sum = 0;
    for(size_t i = 0; i < size(); i++){
        sum += data()[i];
    }
    return sum;
}

std::ostream &operator<<(std::ostream &os, Tensor &obj){
    if(obj.shape()->dims().empty()){
        os << "tensor([])";
        return os;
    }
    std::vector<int> dimensions = obj.shape()->dims();
    os << "tensor(";
    for(auto &dim : dimensions){
        os << "[";
    }
    os << "\n";
    if(dimensions.size() == 1){
        for (size_t i = 0; i < obj.size() - 1; i++){
            os << obj.m_data[i]  << " ";
        }
        os << obj.m_data[obj.size() - 1];
    }else{
        size_t rows = dimensions[dimensions.size() - 2];
        size_t cols = dimensions[dimensions.size() - 1];
        size_t bytesperMat = rows * cols;
        size_t numMatrices = 1;
        for (size_t i = 0; i < dimensions.size() - 2; i++){
            numMatrices *= dimensions[i];
        }
        for (size_t n = 0; n < numMatrices; n++){
            float* matHead = obj.data() + (n * bytesperMat);
            for (size_t i = 0; i < rows; i++){
                for (size_t j = 0; j < cols; j++){
                    os << matHead[i * cols + j] << " ";
                }
                os << "\n";
            }
            os << "\n";
        }
    }
    for(auto &dim : dimensions){
        os << "]";
    }
    os << ")";
    return os;
}
