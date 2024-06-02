#include "binary_primitives.hpp"
#include "matmul.h"
#include "utils.hpp"
#include "tensor.hpp"

namespace core
{
    void Matmul::eval(const std::vector<core::Tensor> &inputs, core::Tensor &out)
    {
        auto &left = inputs[0];
        auto &right = inputs[1];

        std::vector<int> leftShape = left.shape();   //(left.shape());
        std::vector<int> rightShape = right.shape(); // squeezeShape(right.shape());

        if (leftShape.size() < 2 || rightShape.size() < 2)
            throw std::invalid_argument("Matmul currently support only 2 dim tensors");

        size_t M = is_transposed ? leftShape.back() : leftShape.at(leftShape.size() - 2);
        size_t N = is_transposed ? leftShape.at(leftShape.size() - 2) : leftShape.back();
        size_t K;
        if (!is_transposed && leftShape[leftShape.size() - 1] != rightShape[rightShape.size() - 2])
            throw std::invalid_argument("The ncols of the first tensor must be equal nrows! second tensor");

        else if (is_transposed && leftShape[leftShape.size() - 2] != rightShape[rightShape.size() - 2])
            throw std::invalid_argument("The ncols of the first tensor must be equal nrows! second tensor");

        K = rightShape.back();
        std::vector<int> resultShape;
        if (leftShape.size() >= rightShape.size())
            resultShape = leftShape;
        else
            resultShape = rightShape;

        resultShape.at(resultShape.size() - 2) = M;
        resultShape.back() = K;

        int broadCastedDims = 1;
        for (int i = 0; i < resultShape.size() - 2; i++)
            broadCastedDims *= resultShape[i];

        int t1ExtraDims, t2ExtraDims = 1;
        for (int i = 0; i < leftShape.size() - 2; i++)
            t1ExtraDims *= leftShape[i];
        for (int i = 0; i < rightShape.size() - 2; i++)
            t2ExtraDims *= rightShape[i];

        Tensor _left;
        if(!is_transposed){
            _left = left.transpose();
        }


        for (int i = 0; i < broadCastedDims; i++)
        {
            float *t1Data = _left.data() + ((i % t1ExtraDims) * M * N);
            float *t2Data = right.data() + ((i % t2ExtraDims) * M * K);
            float *resultData = out.data() + (i * M * K);

            matmul(M, N, K, t1Data, t2Data, resultData);
        }

    }
}