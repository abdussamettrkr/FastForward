#include "shape.hpp"

int Shape::size(){return this->numel;}

std::vector<int> Shape::dims(){
    std::vector<int> cloneShape(this->shape);
    return cloneShape;
}

std::ostream& operator<<(std::ostream& os, const Shape& obj){
    os << "tensor.Shape([";
    for(auto i = obj.shape.begin(); i < obj.shape.end() - 1; i++){
        os << *i << ", ";
    }
    os << *(obj.shape.end() - 1) << "])";
    return os;
}