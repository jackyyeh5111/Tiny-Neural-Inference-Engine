#pragma once

#include "tensor.h"
#include <string>
#include <vector>
#include <memory>

namespace tnie {

/**
 * @brief Base class for all operators
 */
class Operator {
public:
    explicit Operator(const std::string& name) : name_(name) {}
    virtual ~Operator() = default;

    // Pure virtual method for operator execution
    virtual void forward(const std::vector<Tensor>& inputs, 
                        std::vector<Tensor>& outputs) = 0;
    
    // Operator information
    const std::string& name() const { return name_; }
    virtual std::string type() const = 0;
    
    // Shape inference (optional override)
    virtual std::vector<Shape> infer_output_shapes(
        const std::vector<Shape>& input_shapes) const = 0;

protected:
    std::string name_;
};

/**
 * @brief GEMM (General Matrix Multiply) operator
 * Computes C = alpha * A * B + beta * C
 */
class GemmOp : public Operator {
public:
    GemmOp(const std::string& name, float alpha = 1.0f, float beta = 0.0f, 
           bool trans_a = false, bool trans_b = false)
        : Operator(name), alpha_(alpha), beta_(beta), 
          trans_a_(trans_a), trans_b_(trans_b) {}

    void forward(const std::vector<Tensor>& inputs, 
                std::vector<Tensor>& outputs) override;
    
    std::string type() const override { return "Gemm"; }
    
    std::vector<Shape> infer_output_shapes(
        const std::vector<Shape>& input_shapes) const override;

private:
    float alpha_, beta_;
    bool trans_a_, trans_b_;
    
    void gemm_scalar(const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K);
#ifdef ENABLE_AVX2
    void gemm_avx2(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K);
#endif
};

/**
 * @brief Conv2D operator
 */
class Conv2DOp : public Operator {
public:
    Conv2DOp(const std::string& name, 
             const std::vector<size_t>& kernel_shape,
             const std::vector<size_t>& strides = {1, 1},
             const std::vector<size_t>& pads = {0, 0, 0, 0})
        : Operator(name), kernel_shape_(kernel_shape), 
          strides_(strides), pads_(pads) {}

    void forward(const std::vector<Tensor>& inputs, 
                std::vector<Tensor>& outputs) override;
    
    std::string type() const override { return "Conv2D"; }
    
    std::vector<Shape> infer_output_shapes(
        const std::vector<Shape>& input_shapes) const override;

private:
    std::vector<size_t> kernel_shape_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    
    void conv2d_scalar(const float* input, const float* kernel, float* output,
                      size_t batch, size_t in_channels, size_t out_channels,
                      size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                      size_t kernel_h, size_t kernel_w);
};

/**
 * @brief Operator factory for creating operators
 */
class OperatorFactory {
public:
    static std::unique_ptr<Operator> create_gemm(
        const std::string& name, float alpha = 1.0f, float beta = 0.0f,
        bool trans_a = false, bool trans_b = false);
    
    static std::unique_ptr<Operator> create_conv2d(
        const std::string& name,
        const std::vector<size_t>& kernel_shape,
        const std::vector<size_t>& strides = {1, 1},
        const std::vector<size_t>& pads = {0, 0, 0, 0});
};

} // namespace tnie
