#include "tensor.hpp"
#include <string.h> 

std::vector<int> squeezeVector(std::vector<int> inpVector)
{
    std::vector<int> resultVector;
    for (auto value : inpVector)
    {
        if (value > 1)
            resultVector.push_back(value);
    }
    return resultVector;
}

bool checkBroadcastable(const Tensor &t1, const Tensor &t2)
{
    std::vector<int> t1Dims = t1.shape()->dims();
    std::vector<int> t2Dims = t2.shape()->dims();

    while (t1Dims.size() < t2Dims.size())
        t1Dims.insert(t1Dims.begin(), 1);

    while (t2Dims.size() < t1Dims.size())
        t2Dims.insert(t2Dims.begin(), 1);

    for (int i = 0; i < t1Dims.size() - 2; i++)
    {
        if (t1Dims[i] != t2Dims[i] && (t1Dims[i] != 1 && t2Dims[i] != 1))
        {
            return false;
        }
    }

    return true;
}

std::vector<Tensor> broadCast(const Tensor &t1, const Tensor &t2)
{
    std::vector<int> t1Dims = t1.shape()->dims();
    std::vector<int> t2Dims = t2.shape()->dims();
    std::vector<int> broadCastedDims;
    std::vector<int> mainDims;
    std::vector<int> otherDims;
    int targetSize = -1;
    int sourceSize = -1;
    bool isT1Main = false;
    float *sourceData = nullptr;
    float *targetData = nullptr;
    std::vector<Tensor> result;

    if (t1Dims.size() > t2Dims.size())
    {
        isT1Main = true;
        sourceSize = (*t1.shape()).size();
        sourceData = t2.data();
        mainDims = squeezeVector(t1Dims);
        otherDims = squeezeVector(t2Dims);
        result.push_back(t1);
    }
    else
    {
        sourceSize = (*t2.shape()).size();
        sourceData = t1.data();
        mainDims = squeezeVector(t2Dims);
        otherDims = squeezeVector(t1Dims);
        result.push_back(t2);
    }

    for (int i = 0; i < mainDims.size() - 2; i++)
        broadCastedDims.push_back(mainDims[i]);
    for (int i = 0; i < otherDims.size(); i++)
        broadCastedDims.push_back(otherDims[i]);

    Tensor broadCastedTensor = Tensor::zeros(broadCastedDims);
    targetSize = (*broadCastedTensor.shape()).size();
    targetData = broadCastedTensor.data();

    for (int i = 0; i < targetSize / sourceSize; i++)
    {
        memcpy(targetData + (i * sourceSize), sourceData, sourceSize);
    }
    result.push_back(broadCastedTensor);

    return result;
}