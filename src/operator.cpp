#include "tnie/operator.h"
#include <stdexcept>
#include <algorithm>

#ifdef ENABLE_AVX2
#include <immintrin.h>
#endif

namespace tnie {

// GemmOp implementation
void GemmOp::forward(const std::vector<Tensor>& inputs, 
                    std::vector<Tensor>& outputs) {
    if (inputs.size() < 2) {
        throw std::invalid_argument("GEMM requires at least 2 inputs");
    }
    
    const Tensor& A = inputs[0];
    const Tensor& B = inputs[1];
    
    if (A.dtype() != DataType::FLOAT32 || B.dtype() != DataType::FLOAT32) {
        throw std::invalid_argument("GEMM only supports float32 tensors");
    }
    
    // Get dimensions
    size_t M = trans_a_ ? A.shape()[1] : A.shape()[0];
    size_t K = trans_a_ ? A.shape()[0] : A.shape()[1];
    size_t N = trans_b_ ? B.shape()[0] : B.shape()[1];
    size_t K_B = trans_b_ ? B.shape()[1] : B.shape()[0];
    
    if (K != K_B) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    // Create output tensor if needed
    if (outputs.empty()) {
        outputs.emplace_back(Shape({M, N}), DataType::FLOAT32);
    }
    
    Tensor& C = outputs[0];
    if (C.shape() != Shape({M, N})) {
        throw std::invalid_argument("Output tensor has wrong shape");
    }
    
    // Apply beta scaling to existing output
    if (beta_ != 0.0f) {
        for (size_t i = 0; i < C.size(); ++i) {
            C[i] *= beta_;
        }
    } else {
        C.zero();
    }
    
#ifdef ENABLE_AVX2
    gemm_avx2(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
#else
    gemm_scalar(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
#endif
}

std::vector<Shape> GemmOp::infer_output_shapes(
    const std::vector<Shape>& input_shapes) const {
    if (input_shapes.size() < 2) {
        throw std::invalid_argument("GEMM requires at least 2 inputs");
    }
    
    const Shape& A_shape = input_shapes[0];
    const Shape& B_shape = input_shapes[1];
    
    if (A_shape.ndim() != 2 || B_shape.ndim() != 2) {
        throw std::invalid_argument("GEMM inputs must be 2D");
    }
    
    size_t M = trans_a_ ? A_shape[1] : A_shape[0];
    size_t N = trans_b_ ? B_shape[0] : B_shape[1];
    
    return {Shape({M, N})};
}

void GemmOp::gemm_scalar(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                size_t a_idx = trans_a_ ? (k * M + i) : (i * K + k);
                size_t b_idx = trans_b_ ? (j * K + k) : (k * N + j);
                sum += A[a_idx] * B[b_idx];
            }
            C[i * N + j] += alpha_ * sum;
        }
    }
}

#ifdef ENABLE_AVX2
void GemmOp::gemm_avx2(const float* A, const float* B, float* C,
                      size_t M, size_t N, size_t K) {
    // Simple AVX2 implementation - can be optimized further
    const size_t simd_width = 8; // AVX2 processes 8 floats at once
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += simd_width) {
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t k = 0; k < K; ++k) {
                size_t a_idx = trans_a_ ? (k * M + i) : (i * K + k);
                __m256 a_val = _mm256_broadcast_ss(&A[a_idx]);
                
                size_t remaining = std::min(simd_width, N - j);
                if (remaining == simd_width) {
                    size_t b_base = trans_b_ ? (j * K + k) : (k * N + j);
                    __m256 b_val = _mm256_loadu_ps(&B[b_base]);
                    sum = _mm256_fmadd_ps(a_val, b_val, sum);
                } else {
                    // Handle remaining elements with scalar code
                    for (size_t jj = j; jj < j + remaining; ++jj) {
                        size_t b_idx = trans_b_ ? (jj * K + k) : (k * N + jj);
                        C[i * N + jj] += alpha_ * A[a_idx] * B[b_idx];
                    }
                    break;
                }
            }
            
            if (j + simd_width <= N) {
                __m256 alpha_vec = _mm256_set1_ps(alpha_);
                sum = _mm256_mul_ps(sum, alpha_vec);
                
                __m256 c_val = _mm256_loadu_ps(&C[i * N + j]);
                c_val = _mm256_add_ps(c_val, sum);
                _mm256_storeu_ps(&C[i * N + j], c_val);
            }
        }
    }
}
#endif

// Conv2DOp implementation (basic version)
void Conv2DOp::forward(const std::vector<Tensor>& inputs, 
                      std::vector<Tensor>& outputs) {
    if (inputs.size() < 2) {
        throw std::invalid_argument("Conv2D requires at least 2 inputs (input, kernel)");
    }
    
    const Tensor& input = inputs[0];
    const Tensor& kernel = inputs[1];
    
    // For now, implement basic convolution - can be optimized later
    // Expected input shape: [batch, channels, height, width]
    // Expected kernel shape: [out_channels, in_channels, kernel_h, kernel_w]
    
    if (input.shape().ndim() != 4 || kernel.shape().ndim() != 4) {
        throw std::invalid_argument("Conv2D expects 4D input and kernel tensors");
    }
    
    size_t batch = input.shape()[0];
    size_t in_channels = input.shape()[1];
    size_t in_h = input.shape()[2];
    size_t in_w = input.shape()[3];
    
    size_t out_channels = kernel.shape()[0];
    size_t kernel_in_channels = kernel.shape()[1];
    size_t kernel_h = kernel.shape()[2];
    size_t kernel_w = kernel.shape()[3];
    
    if (in_channels != kernel_in_channels) {
        throw std::invalid_argument("Input and kernel channel dimensions don't match");
    }
    
    // Calculate output dimensions
    size_t out_h = (in_h + pads_[0] + pads_[2] - kernel_h) / strides_[0] + 1;
    size_t out_w = (in_w + pads_[1] + pads_[3] - kernel_w) / strides_[1] + 1;
    
    // Create output tensor if needed
    if (outputs.empty()) {
        outputs.emplace_back(Shape({batch, out_channels, out_h, out_w}), DataType::FLOAT32);
    }
    
    Tensor& output = outputs[0];
    output.zero();
    
    conv2d_scalar(input.data<float>(), kernel.data<float>(), output.data<float>(),
                  batch, in_channels, out_channels, in_h, in_w, out_h, out_w, kernel_h, kernel_w);
}

std::vector<Shape> Conv2DOp::infer_output_shapes(
    const std::vector<Shape>& input_shapes) const {
    if (input_shapes.size() < 2) {
        throw std::invalid_argument("Conv2D requires at least 2 inputs");
    }
    
    const Shape& input_shape = input_shapes[0];
    const Shape& kernel_shape = input_shapes[1];
    
    size_t batch = input_shape[0];
    size_t in_h = input_shape[2];
    size_t in_w = input_shape[3];
    size_t out_channels = kernel_shape[0];
    
    size_t out_h = (in_h + pads_[0] + pads_[2] - kernel_shape_[0]) / strides_[0] + 1;
    size_t out_w = (in_w + pads_[1] + pads_[3] - kernel_shape_[1]) / strides_[1] + 1;
    
    return {Shape({batch, out_channels, out_h, out_w})};
}

void Conv2DOp::conv2d_scalar(const float* input, const float* kernel, float* output,
                           size_t batch, size_t in_channels, size_t out_channels,
                           size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                           size_t kernel_h, size_t kernel_w) {
    // Basic implementation - can be optimized with better memory access patterns
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * strides_[0] + kh - pads_[0];
                                int iw = ow * strides_[1] + kw - pads_[1];
                                
                                if (ih >= 0 && ih < static_cast<int>(in_h) && 
                                    iw >= 0 && iw < static_cast<int>(in_w)) {
                                    size_t input_idx = b * in_channels * in_h * in_w + 
                                                     ic * in_h * in_w + ih * in_w + iw;
                                    size_t kernel_idx = oc * in_channels * kernel_h * kernel_w + 
                                                      ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    
                    size_t output_idx = b * out_channels * out_h * out_w + 
                                      oc * out_h * out_w + oh * out_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// OperatorFactory implementation
std::unique_ptr<Operator> OperatorFactory::create_gemm(
    const std::string& name, float alpha, float beta, bool trans_a, bool trans_b) {
    return std::make_unique<GemmOp>(name, alpha, beta, trans_a, trans_b);
}

std::unique_ptr<Operator> OperatorFactory::create_conv2d(
    const std::string& name, const std::vector<size_t>& kernel_shape,
    const std::vector<size_t>& strides, const std::vector<size_t>& pads) {
    return std::make_unique<Conv2DOp>(name, kernel_shape, strides, pads);
}

} // namespace tnie
