#include <vector>

class Tensor{
    public:
        static Tensor ones(unsigned int rows, unsigned int cols);
        static Tensor zeros(unsigned int rows, unsigned int cols);
        //static Tensor randn(unsigned int rows, unsigned int cols);
        

        float* data() const;

        //Shape* shape(); # Must be implemented 

        
        Tensor operator +(const Tensor &other);
        template <typename T>
        Tensor operator +(T value);
        Tensor operator -(const Tensor &other);
        template <typename T>
        Tensor operator -(T value);
        Tensor operator /(const Tensor &other);
        template <typename T>
        Tensor operator /(T value);
        Tensor operator *(const Tensor &other);
        template <typename T>
        Tensor operator *(T value);


    private:
        float* m_data;  
        std::vector <unsigned int> m_shape;
        unsigned int m_rows;
        unsigned int m_cols;
        unsigned int n_dim;

        Tensor(unsigned int rows, unsigned int cols, float* data);

        int size();


};

Tensor::Tensor(std::vector <unsigned int> shape){
    std::vector<unsigned int> m_shape(shape);
    unsigned int total_size = 0;
    for (int i=0; i < m_shape.size(); i++)
    {
        total_size += m_shape[i];
    }

    m_data = new float[total_size];    
}

Tensor::Tensor(unsigned int rows, unsigned int cols, float* data){
    m_data = data;

    m_rows = rows;
    m_cols = cols;
}

float* Tensor::data() const{
    return m_data;
}

int Tensor::size(){
    return m_rows*m_cols;
}

Tensor Tensor::ones(unsigned int rows, unsigned int cols){
    float* data = new float(rows*cols);
    for(int i = 0; i < rows*cols; i ++)
        *(data+i) = 1;
    return Tensor(rows, cols, data);
}


Tensor Tensor::zeros(unsigned int rows, unsigned int cols){
    float* data = new float(rows*cols);
    for(int i = 0; i < rows*cols; i ++)
        *(data+i) = 0;

    return Tensor(rows, cols, data);
}

// We will perform operations inplace!
Tensor Tensor::operator+(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] += other.data()[i];
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator+(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] += value;
    }

    return *this;
}


Tensor Tensor::operator-(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] -= other.data()[i];
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator-(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] -= value;
    }

    return *this;
}

Tensor Tensor::operator*(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] *= other.data()[i];
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator*(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] *= value;
    }

    return *this;
}

Tensor Tensor::operator/(const Tensor &other){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] /= other.data()[i];
    }

    return *this;
}

template <typename T>
Tensor Tensor::operator/(T value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] /= value;
    }

    return *this;
}
