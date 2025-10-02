#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <cstddef>

namespace tnie {

/**
 * @brief Tensor shape representation
 */
class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
    explicit Shape(const std::vector<size_t>& dims) : dims_(dims) {}

    size_t ndim() const { return dims_.size(); }
    size_t size() const;
    size_t operator[](size_t idx) const { return dims_[idx]; }
    
    const std::vector<size_t>& dims() const { return dims_; }
    
    bool operator==(const Shape& other) const { return dims_ == other.dims_; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    std::vector<size_t> dims_;
};

/**
 * @brief Data type enumeration
 */
enum class DataType {
    FLOAT32,
    INT32,
    // Add more types as needed
};

/**
 * @brief Minimal tensor implementation
 */
class Tensor {
public:
    Tensor() = default;
    Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32);
    Tensor(const Shape& shape, const std::vector<float>& data);
    
    // Copy and move constructors/assignment
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor() = default;

    // Accessors
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    size_t size() const { return shape_.size(); }
    size_t size_bytes() const;
    
    // Data access
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }
    
    template<typename T>
    T* data() { return static_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_.get()); }
    
    // Convenience methods for float tensors
    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;
    
    // Utility methods
    void fill(float value);
    void zero();
    
    // Factory methods
    static Tensor zeros(const Shape& shape, DataType dtype = DataType::FLOAT32);
    static Tensor ones(const Shape& shape, DataType dtype = DataType::FLOAT32);
    static Tensor random(const Shape& shape, DataType dtype = DataType::FLOAT32);

private:
    Shape shape_;
    DataType dtype_ = DataType::FLOAT32;
    std::unique_ptr<void, void(*)(void*)> data_{nullptr, [](void*){}};
    
    void allocate();
    size_t type_size() const;
};

} // namespace tnie
