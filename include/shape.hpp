#include <vector>
#include <iostream>

class Shape{
    private:
        std::vector<int> shape;
        int ndims;
        int numel;
    
    public:
        template <typename Iterable>
        Shape(Iterable &arraylike){
            this->ndims = 0;
            this->numel = 1;
            for(auto &dim : arraylike){
                shape.push_back(dim);
                ndims++;
                numel *= dim;
            }
        }
        std::vector<int> dims();
        int size();
        friend std::ostream& operator<<(std::ostream& os, const Shape& obj);
};