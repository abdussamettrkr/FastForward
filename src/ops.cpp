#include "ops.hpp"
#include "copy.hpp"


namespace ops{
core::Tensor unary_op(const core::Tensor& in, core::Primitive& op){    
    auto out = core::Tensor(in.shape(), new float[in.size()]);
    op.eval({in}, out);
    return out;
}


core::Tensor binary_op(const core::Tensor& left, const core::Tensor& right, core::Primitive& op){
    if(!checkBroadcastable(left.shape(), right.shape()))
        throw std::logic_error("Tensors are not compatiable!");
    
    auto bshape = broadcastShapes(left.shape(), right.shape());
    auto bleft = broadcastTo(left, bshape);
    auto bright = broadcastTo(right, bshape);
    size_t out_size = 1;
    for (auto dim : bshape)
        out_size *= dim;
    
    auto out = core::Tensor(bshape, new float[out_size]);
    op.eval({bleft, bright}, out);
    return out;
}


core::Tensor add(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Add op;
    return binary_op(left, right, op);
}

core::Tensor substract(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Substract op;
    return binary_op(left, right, op);
}

core::Tensor divide(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Divide op;
    return binary_op(left, right, op);
}

core::Tensor multiply(const core::Tensor& left, const core::Tensor& right){
    // We are going to make it lazy
    core::Divide op;
    return binary_op(left, right, op);
}

core::Tensor log(const core::Tensor& in){
    core::Log op;
    return unary_op(in, op);
}

core::Tensor sqrt(const core::Tensor& in){
    core::Sqrt op;
    return unary_op(in, op);
}

core::Tensor exp(const core::Tensor& in){
    core::Exp op;
    return unary_op(in, op);
}

core::Tensor relu(const core::Tensor& in){
    core::Relu op;
    return unary_op(in, op);
}

core::Tensor matmul(const core::Tensor& left, const core::Tensor& right){
    const std::vector<int>& leftShape = left.shape();
    const std::vector<int>& rightShape = right.shape();
    std::vector<int> leftBaseShape(leftShape.begin(), leftShape.end()-2);
    std::vector<int> rightBaseShape(rightShape.begin(), rightShape.end()-2);

    
    auto outShape = broadcastShapes(leftBaseShape, rightBaseShape);
    outShape.push_back(left.shape()[left.ndim()-2]);
    outShape.push_back(right.shape()[right.ndim()-1]);

    auto out = core::Tensor(outShape);

    core::Matmul op;
    op.eval({left, right}, out);   
    return out;
}

core::Tensor conv2d(const core::Tensor& input, const core::Tensor& kernel, size_t stride){
    // TODO: Check shapes are compatiable
    const std::vector<int>& inputShape = input.shape();
    const std::vector<int>& kernelShape = kernel.shape();

    std::vector<int> outShape(inputShape.begin(), inputShape.end()-1);
    outShape.insert(outShape.end(), kernelShape[0]);
    outShape[1] = (outShape[1] - kernelShape[1]) / stride + 1;
    outShape[2] = (outShape[2] - kernelShape[2]) / stride + 1;

    auto out = core::Tensor(outShape);
    core::Convolution op(stride);
    op.eval({input, kernel}, out);
    return out;
}

core::Tensor maxpool2d(const core::Tensor& input, size_t kernel_size, size_t stride){
    // TODO: Check shapes are compatiable
    const std::vector<int>& inputShape = input.shape();

    std::vector<int> outShape(inputShape.begin(), inputShape.end());
    outShape[1] = (outShape[1] - kernel_size) / stride + 1;
    outShape[2] = (outShape[2] - kernel_size) / stride + 1;

    auto out = core::Tensor(outShape);
    core::MaxPool2D op(kernel_size, stride);
    op.eval({input}, out);
    return out;
}

core::Tensor im2col(const core::Tensor& input, size_t kernel_h, size_t kernel_w, size_t padding, size_t stride){
    if(padding != 0){
        throw std::logic_error("Currently im2col only supports padding=0");
    }
    if(stride != 1){
        throw std::logic_error("Currently im2col only supports stride=1");
    }
    std::vector<int> inShapes = input.shape();
    std::vector<int> inStrides = input.strides();
    int nB = inShapes[0];
    int iH = inShapes[1];
    int iW = inShapes[2];
    int nC = inShapes[3];
    int inStrideH = inStrides[1];
    int inStrideW = inStrides[2];

    if(nB != 1){
        throw std::logic_error("Currently im2col only supports batchsize=1");
    }

    int oH = (iH - kernel_h) / stride + 1;
    int oW = (iW - kernel_w) / stride + 1;

    auto out  = core::Tensor({static_cast<int>(nC*kernel_h*kernel_w), static_cast<int>(oH*oW)});
    const float *inData = input.data();
    float *outData = out.data();

    for (size_t output_y = 0; output_y < oH; output_y++)
    {
        for (size_t output_x = 0; output_x < oW; output_x++)
        {
            for (size_t kernel_y = 0; kernel_y < kernel_h; kernel_y++)
            {
                const size_t input_y = output_y + kernel_y;
                if (input_y < iH)
                {
                    for (size_t kernel_x = 0; kernel_x < kernel_w; kernel_x++)
                    {
                        const size_t input_x = output_x + kernel_x;
                        if (input_x < iW)
                        {
                            memcpy(outData, inData + input_y * inStrideH + input_x * inStrideW, nC* sizeof(float));
                        }
                        else
                        {
                            memset(outData, 0, nC);
                        }
                        outData = outData + nC;
                    }
                }
                else
                {
                    memset(outData, 0, kernel_w * nC);
                    outData = outData + kernel_w * nC;
                }
            }
        }
    }
    return out;
}

core::Tensor pad(const core::Tensor& input, std::vector<int> pad_width){
    if(input.shape().size() != pad_width.size()){
        throw std::logic_error("Pad with size must match with input shape size");
    }

    std::vector<int> result_shape;
    for (size_t i = 0; i < pad_width.size(); i++)
    {
        result_shape.push_back(input.shape()[i] + pad_width[i]*2);
    }

    auto result = core::Tensor(result_shape);
    size_t offset = 0;
    for (size_t i = 0; i < pad_width.size(); i++)
    {
        offset += pad_width[i] * result.strides()[i];
    }
    
    copy(input.data(), result.data(), input.size(), 0, offset, input.shape(), result.strides());
    return result;
}

core::Tensor softmax(const core::Tensor& input, int axis){
    if(axis < 0)
        axis = input.ndim() + axis;
    
    auto out = new core::Tensor(input.shape());
    
    core::Softmax op(axis);
    op.eval({input}, *out);
    return *out;
}

core::Tensor reduce(const core::Tensor&input, const std::vector<int>& axes, bool keepdims, ReductionType type){
    core::Primitive* op = nullptr;
    core::Tensor* out;
    ReductionMethod reduction_method;
    std::vector<int> result_shape;
    //All reduce
    if (input.is_contiguous() && (input.shape().size() == axes.size() || axes.size() == 0)){
        reduction_method = ReductionMethod::ContiguousAllReduce;
        if (!keepdims)
            result_shape = {};
        else{
            for (size_t i = 0; i < input.shape().size(); i++)
                result_shape.push_back(1);
        }
    }
    else{
        reduction_method = ReductionMethod::ContiguousReduce;
        std::vector<int> out_shape;
        for (int i =0; i < input.shape().size(); i++)
        {
            bool isIn = false;
            for (auto axis: axes)
            {
                isIn = isIn | (i==axis);
            }
            
            if (!isIn)
                out_shape.push_back(input.shape()[i]);
            else if(keepdims)
                out_shape.push_back(1);
        }
        result_shape = out_shape;
    }
    
    if(type == ReductionType::MAX){
        op = new core::Max(reduction_method, axes);
    }
    else if(type==ReductionType::SUM){
        op = new core::Sum(reduction_method, axes);
    }
    else if(type==ReductionType::MIN){
        op = new core::Min(reduction_method, axes);
    }
    else if(type==ReductionType::PROD){
        op = new core::Prod(reduction_method, axes);
    }
    out = new core::Tensor(result_shape);


    op->eval({input}, *out);
    return *out;
}

core::Tensor max(const core::Tensor&input, const std::vector<int>& axes, bool keepdims){
    return reduce(input, axes, keepdims, ReductionType::MAX);
}

core::Tensor sum(const core::Tensor&input, const std::vector<int>& axes, bool keepdims){
    return reduce(input, axes, keepdims, ReductionType::SUM);
}

core::Tensor min(const core::Tensor&input, const std::vector<int>& axes, bool keepdims){
    return reduce(input, axes, keepdims, ReductionType::MIN);
}

core::Tensor prod(const core::Tensor&input, const std::vector<int>& axes, bool keepdims){
    return reduce(input, axes, keepdims, ReductionType::PROD);
}
}