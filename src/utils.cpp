#include "tensor.hpp"

std::vector<int> squeezeShape(std::vector<int> inputShape)
{
    std::vector<int> resultShape;
    for (auto value : inputShape)
    {
        if (value > 1)
            resultShape.push_back(value);
    }
    return resultShape;
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
            return false;
    }

    return true;
}