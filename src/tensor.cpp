#include "tnie/tensor.h"
#include <cstring>
#include <stdexcept>
#include <random>
#include <algorithm>

namespace tnie {

// Shape implementation
size_t Shape::size() const {
    if (dims_.empty()) return 0;
    size_t total = 1;
    for (size_t dim : dims_) {
        total *= dim;
    }
    return total;
}

// Tensor implementation
Tensor::Tensor(const Shape& shape, DataType dtype) 
    : shape_(shape), dtype_(dtype) {
    allocate();
}

Tensor::Tensor(const Shape& shape, const std::vector<float>& data)
    : shape_(shape), dtype_(DataType::FLOAT32) {
    if (data.size() != shape.size()) {
        throw std::invalid_argument("Data size doesn't match tensor shape");
    }
    allocate();
    std::memcpy(this->data(), data.data(), data.size() * sizeof(float));
}

Tensor::Tensor(const Tensor& other) 
    : shape_(other.shape_), dtype_(other.dtype_) {
    allocate();
    std::memcpy(data(), other.data(), size_bytes());
}

Tensor::Tensor(Tensor&& other) noexcept 
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), 
      data_(std::move(other.data_)) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        allocate();
        std::memcpy(data(), other.data(), size_bytes());
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = std::move(other.data_);
    }
    return *this;
}

size_t Tensor::size_bytes() const {
    return size() * type_size();
}

float& Tensor::operator[](size_t idx) {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("Tensor is not float32 type");
    }
    if (idx >= size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return static_cast<float*>(data())[idx];
}

const float& Tensor::operator[](size_t idx) const {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("Tensor is not float32 type");
    }
    if (idx >= size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return static_cast<const float*>(data())[idx];
}

void Tensor::fill(float value) {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("Fill only supported for float32 tensors");
    }
    float* ptr = data<float>();
    std::fill(ptr, ptr + size(), value);
}

void Tensor::zero() {
    std::memset(data(), 0, size_bytes());
}

Tensor Tensor::zeros(const Shape& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.zero();
    return tensor;
}

Tensor Tensor::ones(const Shape& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    if (dtype == DataType::FLOAT32) {
        tensor.fill(1.0f);
    } else {
        throw std::runtime_error("Ones only supported for float32 tensors");
    }
    return tensor;
}

Tensor Tensor::random(const Shape& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    if (dtype == DataType::FLOAT32) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        float* ptr = tensor.data<float>();
        for (size_t i = 0; i < tensor.size(); ++i) {
            ptr[i] = dis(gen);
        }
    } else {
        throw std::runtime_error("Random only supported for float32 tensors");
    }
    return tensor;
}

void Tensor::allocate() {
    if (size() == 0) return;
    
    size_t bytes = size_bytes();
    void* ptr = std::aligned_alloc(32, bytes); // 32-byte alignment for AVX2
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    data_ = std::unique_ptr<void, void(*)(void*)>(ptr, [](void* p) { std::free(p); });
}

size_t Tensor::type_size() const {
    switch (dtype_) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        default: throw std::runtime_error("Unknown data type");
    }
}

} // namespace tnie
