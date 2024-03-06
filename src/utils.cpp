#include "utils.hpp"

using namespace core;

std::vector<int> squeezeShape(const std::vector<int> inputShape)
{
    std::vector<int> resultShape;
    for (auto value : inputShape)
    {
        if (value > 1)
            resultShape.push_back(value);
    }
    return resultShape;
}

bool checkBroadcastable(const Tensor t1, const Tensor t2)
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

std::vector<int> broadcastShapes(const std::vector<int>& shape1, const std::vector<int>& shape2){
    int diff = shape1.size() - shape2.size();
    const auto& big = shape1.size() > shape2.size() ? shape1 : shape2;
    const auto& small = shape1.size() <= shape2.size() ? shape1 : shape2;
    std::vector<int> resultShape(big.size());

    if (diff < 0)
        diff = -diff;

    for (size_t i = 0; i < big.size(); i++)
    {   
        if (i < diff){
            resultShape[i] = big[i];
            continue;
        }
        if(big[i] == 1 || small[i - diff] == 1)
            resultShape[i] = big[i] * small[i - diff];
        else if(big[i] != small[i - diff])
            throw std::logic_error("Provided shapes cannot be broadcasted");
        else
            resultShape[i] = big[i];
    }
    return resultShape;
}

std::vector<int> calculateStride(const std::vector<int> shape){
    size_t prod = 1 * 8;
    std::vector<int> strides = std::vector<int>(shape.size(),0);
    for (int i = shape.size()-1; i >= 0; i--) {
        strides[i] = prod;
        prod *= shape[i];
    }

    return strides;
}