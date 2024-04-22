#include "binary_primitives.hpp"
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

        size_t M = leftShape.at(leftShape.size() - 2);
        size_t N = leftShape.back();
        size_t K;
        if (!is_transposed){
            if (leftShape[leftShape.size() - 1] != rightShape[rightShape.size() - 2])
                throw std::invalid_argument("The ncols of the first tensor must be equal nrows! second tensor");

            K = rightShape.back();
        }
        else{
            K = rightShape.at(rightShape.size() - 2);
        }

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


        for (int i = 0; i < broadCastedDims; i++)
        {
            float *t1Data = left.data() + ((i % t1ExtraDims) * M * N);
            float *t2Data = right.data() + ((i % t2ExtraDims) * M * K);
            float *resultData = out.data() + (i * M * K);

            for (int row = 0; row < M; row++)
                for (int col = 0; col < K; col++)
                    for (int inner = 0; inner < N; inner++)
                    {
                        if(is_transposed)
                            resultData[row * K + col] += t1Data[row * N + inner] * t2Data[K * col + inner];
                        else
                            resultData[row * K + col] += t1Data[row * N + inner] * t2Data[K * inner + col];
                    }
        }
    }
}