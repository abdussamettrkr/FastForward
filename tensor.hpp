class Tensor{
    public:
        Tensor(unsigned int rows, unsigned int cols);        

        static Tensor ones(unsigned int rows, unsigned int cols);
        static Tensor zeros(unsigned int rows, unsigned int cols);
        //static Tensor randn(unsigned int rows, unsigned int cols);
        

        float* data() const;

        //Shape* shape(); # Must be implemented 

        
        Tensor operator +(const Tensor &other);
        Tensor operator +(const float value);
        Tensor operator +(const int value);
        Tensor operator -(const Tensor &other);
        Tensor operator -(const float value);
        Tensor operator -(const int value);
        Tensor operator /(const Tensor &other);
        Tensor operator /(const float value);
        Tensor operator /(const int value);
        Tensor operator *(const Tensor &other);
        Tensor operator *(const float value);
        Tensor operator *(const int value);


    private:
        float* m_data;  
        unsigned int m_rows;
        unsigned int m_cols;

        Tensor(unsigned int rows, unsigned int cols, float* data);

        int size();


};

Tensor::Tensor(unsigned int rows, unsigned int cols){
    m_data = new float[rows*cols];

    m_rows = rows;
    m_cols = cols;
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

Tensor Tensor::operator+(const int value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] += value;
    }

    return *this;
}

Tensor Tensor::operator+(const float value){
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

Tensor Tensor::operator-(const int value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] -= value;
    }

    return *this;
}

Tensor Tensor::operator-(const float value){
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

Tensor Tensor::operator*(const int value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] *= value;
    }

    return *this;
}

Tensor Tensor::operator*(const float value){
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

Tensor Tensor::operator/(const int value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] /= value;
    }

    return *this;
}

Tensor Tensor::operator/(const float value){
    int total_size = size();
    for(int i = 0; i < total_size; i++){
        m_data[i] /= value;
    }

    return *this;
}