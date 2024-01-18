#include "tensor.hpp"

class Conv2D{
    private:
        Tensor kernel;
        int padding;
        int kernel_size;
        int stride;
        Tensor pad(Tensor &tensor);
        std::string paddingMode;
    public:
        Conv2D(int kernel_size, int stride, int padding);
        Tensor operator()(Tensor &applyOn);
};

Conv2D::Conv2D(int kernel_size, int stride, int padding){
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;
    kernel = Tensor::ones({kernel_size, kernel_size});
    this->paddingMode = paddingMode;
}

Tensor Conv2D::pad(Tensor &tensor){
    // Zero pads only for now
    Shape* shape = tensor.shape();
    std::vector<int> dims = shape->dims(); 
    int rows = *(dims.end() - 2);
    int cols = *(dims.end() - 1);
    *(dims.end() - 1) = cols + 2 * padding;
    *(dims.end() - 2) = rows + 2 * padding;
    size_t paddedRows = *(dims.end() - 2);
    size_t paddedCols = *(dims.end() - 1);

    
    Tensor paddedTensor = Tensor::zeros(dims);
    float* paddedData = paddedTensor.data();
    float* tensorData = tensor.data();

    size_t bytesToNextMat = rows * cols;
    size_t bytesToNextPaddedMat = paddedRows * paddedCols;
    size_t numMatrices = 1;
    for(auto i = 0; i < dims.size() - 2; i++){
        numMatrices *= dims[i];
    }
    for(auto n = 0; n < numMatrices; n++){
        float* tensorHead = tensorData + n * bytesToNextMat;
        float* paddedHead = paddedData + n * bytesToNextPaddedMat;
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                *(paddedHead + (i + padding) * paddedCols + j + padding) = *(tensorHead + i * cols + j);
            }
        }
    }
    return paddedTensor;
}

Tensor Conv2D::operator()(Tensor &applyOn){
    // The default approach is applying the kernel on last 2 dimensions
    Tensor padded = pad(applyOn);
    std::vector<int> outShape = applyOn.shape()->dims();
    Tensor out = Tensor::zeros(outShape);
    float* data = padded.data();
    float tempSlice[kernel_size * kernel_size];
    
    Shape* paddedShape = padded.shape();
    std::vector<int> dims = paddedShape->dims();
    int rows = *(dims.end() - 2);
    int cols = *(dims.end() - 1);

    size_t bytesToNextMat = rows * cols;
    size_t bytesToNextOut = (rows - 2 * padding) * (cols - 2 * padding); 
    size_t numMatrices = 1;
    for(auto i = 0; i < dims.size() - 2; i++){
        numMatrices *= dims[i];
    }
    for(auto n = 0; n < numMatrices; n++){
        float* matrixHead = data + n * bytesToNextMat;
        float* outHead = out.data() + n * bytesToNextOut;
        for(size_t i = padding; i < rows - padding; i++){
            for(size_t j = padding; j < cols - padding; j++){
                Tensor slice = padded.getKernel(tempSlice, matrixHead, kernel_size, i, j);
                outHead[(i - padding) * (rows - 2 * padding) + j - padding] = (slice * kernel).sum();
            }
        }
    }
    return out;
}