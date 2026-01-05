#include "ops.h"

#include "utils.h"

namespace tiny_engine::ops {

    TensorPtr relu(const std::vector<const onnx::TensorProto*>& inputs,
                   const onnx::NodeProto& node) {
        auto output = std::make_unique<onnx::TensorProto>(*inputs[0]);
        float* data = reinterpret_cast<float*>(output->mutable_raw_data()->data());
        int size = output->raw_data().size() / sizeof(float);

        for (int i = 0; i < size; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
        return output;
    }

    /*
       Say 4D input tensor (often used for Images: Batch, Channels, Height, Width)
       with dimensions [2, 3, 4, 5] and the attribute axis = 1.

       d1 = 2 (Batch)
       d2 = 3*4*5 = 60

       output: [2, 60] because 3*4*5=60
    */
    TensorPtr flatten(const std::vector<const onnx::TensorProto*>& inputs,
                      const onnx::NodeProto& node) {
        auto output = std::make_unique<onnx::TensorProto>(*inputs[0]);
        int axis = 1;
        for (auto& attr : node.attribute())
            if (attr.name() == "axis")
                axis = attr.i();

        int64_t d1 = 1, d2 = 1;
        for (int i = 0; i < output->dims_size(); ++i) {
            if (i < axis)
                d1 *= output->dims(i);
            else
                d2 *= output->dims(i);
        }

        output->clear_dims();
        output->add_dims(d1);
        output->add_dims(d2);
        return output;
    }

    TensorPtr gemm(const std::vector<const onnx::TensorProto*>& inputs,
                   const onnx::NodeProto& node) {
        if (inputs.size() < 2) {
            throw std::invalid_argument("Gemm requires at least 2 inputs (A, B)");
        }

        // 1. Parse Attributes
        // ONNX Gemm: alpha * AB + beta * C
        float alpha = 1.0f, beta = 1.0f;
        int transA = 0, transB = 0;
        for (const auto& attr : node.attribute()) {
            if (attr.name() == "alpha")
                alpha = attr.f();
            else if (attr.name() == "beta")
                beta = attr.f();
            else if (attr.name() == "transA")
                transA = attr.i();
            else if (attr.name() == "transB")
                transB = attr.i();
        }

        const auto* A = inputs[0];
        const auto* B = inputs[1];

        // 2. Determine Dimensions
        int64_t M = transA ? A->dims(1) : A->dims(0);
        int64_t K = transA ? A->dims(0) : A->dims(1);  // Inner dimension
        int64_t N = transB ? B->dims(0) : B->dims(1);

        // 3. Prepare Data Pointers (handling transposes if necessary)
        const float* a_ptr = reinterpret_cast<const float*>(A->raw_data().data());
        const float* b_ptr = reinterpret_cast<const float*>(B->raw_data().data());

        // TODO: improve transpose
        // Simple transpose helper for B (most common case in FC layers)
        std::vector<float> a_processed;
        if (transA) {
            a_processed.resize(M * K);
            int orig_rows = A->dims(0);
            int orig_cols = A->dims(1);
            for (int r = 0; r < orig_rows; ++r)
                for (int c = 0; c < orig_cols; ++c)
                    a_processed[c * orig_rows + r] = a_ptr[r * orig_cols + c];
            a_ptr = a_processed.data();
        }

        // Simple transpose helper for B (most common case in FC layers)
        std::vector<float> b_processed;
        if (transB) {
            b_processed.resize(K * N);
            int orig_rows = B->dims(0);
            int orig_cols = B->dims(1);
            for (int r = 0; r < orig_rows; ++r)
                for (int c = 0; c < orig_cols; ++c)
                    b_processed[c * orig_rows + r] = b_ptr[r * orig_cols + c];
            b_ptr = b_processed.data();
        }

        // 4. Handle Bias (C)
        // ONNX Gemm: alpha * AB + beta * C
        std::vector<float> c_data(M * N, 0.0f);
        if (inputs.size() == 3) {
            const auto* C = inputs[2];
            const float* c_ptr = reinterpret_cast<const float*>(C->raw_data().data());

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    // Handle broadcast of 1D bias or full 2D bias
                    int c_idx = (C->dims_size() == 1) ? j : (i * N + j);
                    c_data[i * N + j] = c_ptr[c_idx];
                }
            }
        }

        // 5. Execute Math
        std::vector<float> out_data(M * N);
        // Call your raw gemm function from gemm.h
        _gemm(a_ptr, b_ptr, c_data.data(), out_data.data(), M, K, N, alpha, beta);

        // 6. Wrap in TensorProto
        auto result = std::make_unique<onnx::TensorProto>();
        result->set_name(node.output(0));
        result->set_data_type(onnx::TensorProto::FLOAT);
        result->add_dims(M);
        result->add_dims(N);
        result->set_raw_data(out_data.data(), out_data.size() * sizeof(float));

        return result;
    }

    // Helper to get attributes with defaults
    int64_t get_attr(const onnx::NodeProto& node, const std::string& name, int64_t default_val) {
        for (const auto& attr : node.attribute()) {
            if (attr.name() == name)
                return attr.i();
        }
        return default_val;
    }

    std::vector<int64_t> get_attrs(const onnx::NodeProto& node,
                                   const std::string& name,
                                   int64_t count,
                                   int64_t default_val) {
        for (const auto& attr : node.attribute()) {
            if (attr.name() == name) {
                return {attr.ints().begin(), attr.ints().end()};
            }
        }
        return std::vector<int64_t>(count, default_val);
    }

    TensorPtr conv(const std::vector<const onnx::TensorProto*>& inputs,
                   const onnx::NodeProto& node) {
        const auto* X = inputs[0];
        const auto* W = inputs[1];
        const auto* B = (inputs.size() > 2) ? inputs[2] : nullptr;

        // 1. Get Attributes
        auto strides = get_attrs(node, "strides", 2, 1);      // {sH, sW}
        auto pads = get_attrs(node, "pads", 4, 0);            // {t, l, b, r}
        auto dilations = get_attrs(node, "dilations", 2, 1);  // {dH, dW}

        int64_t N = X->dims(0), C = X->dims(1), H = X->dims(2), W_in = X->dims(3);
        int64_t M = W->dims(0), kH = W->dims(2), kW = W->dims(3);

        // 2. Calculate Output Dimensions
        int64_t H_out = (H + pads[0] + pads[2] - (dilations[0] * (kH - 1) + 1)) / strides[0] + 1;
        int64_t W_out =
                (W_in + pads[1] + pads[3] - (dilations[1] * (kW - 1) + 1)) / strides[1] + 1;

        const float* x_ptr = reinterpret_cast<const float*>(X->raw_data().data());
        const float* w_ptr = reinterpret_cast<const float*>(W->raw_data().data());
        const float* b_ptr = B ? reinterpret_cast<const float*>(B->raw_data().data()) : nullptr;

        std::vector<float> final_out(N * M * H_out * W_out);

        // 3. Process Batch
        // im2col size: [C * kH * kW] x [H_out * W_out]
        int64_t k_size = C * kH * kW;
        int64_t n_patches = H_out * W_out;
        std::vector<float> col_data(k_size * n_patches);

        for (int64_t n = 0; n < N; ++n) {
            // --- im2col Step ---
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t row = (c * kH * kW) + (kh * kW) + kw;
                        for (int64_t oh = 0; oh < H_out; ++oh) {
                            for (int64_t ow = 0; ow < W_out; ++ow) {
                                int64_t h_in = oh * strides[0] + kh * dilations[0] - pads[0];
                                int64_t w_in = ow * strides[1] + kw * dilations[1] - pads[1];

                                float val = 0;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W_in) {
                                    val = x_ptr[n * (C * H * W_in) + c * (H * W_in) + h_in * W_in +
                                                w_in];
                                }
                                col_data[row * n_patches + (oh * W_out + ow)] = val;
                            }
                        }
                    }
                }
            }

            // --- GEMM Step ---
            // Weight: [M x k_size], Col: [k_size x n_patches] -> Out: [M x n_patches]
            float* batch_out_ptr = &final_out[n * M * n_patches];

            // We use alpha=1.0. Bias is handled after or via beta=1.0 if we broadcast B
            // Note: Your _gemm expects C to be the same size as Out.
            // We'll manually handle bias for efficiency since it's just a 1D vector.
            std::fill(batch_out_ptr, batch_out_ptr + (M * n_patches), 0.0f);

            _gemm(w_ptr,
                  col_data.data(),
                  nullptr,
                  batch_out_ptr,
                  M,
                  k_size,
                  n_patches,
                  1.0f,
                  0.0f);

            // --- Apply Bias ---
            if (b_ptr) {
                for (int m = 0; m < M; ++m) {
                    for (int p = 0; p < n_patches; ++p) {
                        batch_out_ptr[m * n_patches + p] += b_ptr[m];
                    }
                }
            }
        }

        // 4. Wrap result
        auto result = std::make_unique<onnx::TensorProto>();
        result->set_data_type(onnx::TensorProto::FLOAT);
        for (auto d : {N, M, H_out, W_out})
            result->add_dims(d);
        result->set_raw_data(final_out.data(), final_out.size() * sizeof(float));
        return result;
    }

    /**
     * @brief Performs General Matrix Multiplication with bias: out = A * B + C
     *
     * This function implements GEMM (General Matrix Multiplication) which is fundamental
     * for neural network operations, particularly fully connected layers.
     *
     * @param A Input matrix A with dimensions (n x m) in row-major order
     * @param B Input matrix B with dimensions (m x k) in row-major order
     * @param C Bias vector with dimensions (n x k) in row-major order
     * @param out Output matrix with dimensions (n x k) in row-major order
     * @param n Number of rows in A and output
     * @param m Number of columns in A and rows in B
     * @param k Number of columns in B and output
     */
    void _gemm(const float* A,
               const float* B,
               const float* C,
               float* out,
               int n,
               int m,
               int k,
               float alpha,
               float beta) {

        // cache friendly matmul: out = A * B
        for (int r = 0; r < n; ++r) {
            for (int i = 0; i < m; ++i) {
                float a = A[r * m + i];
                for (int c = 0; c < k; ++c) {
                    out[r * k + c] += a * B[i * k + c];
                }
            }
        }

        // No bias, return
        if (C == nullptr)
            return;

        // onnx Gemm: alpha * AB + beta * C
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; ++c) {
                out[r * k + c] = alpha * out[r * k + c] + beta * C[r * k + c];
            }
        }
    }

    TensorPtr maxpool(const std::vector<const onnx::TensorProto*>& inputs,
                      const onnx::NodeProto& node) {
        const auto* X = inputs[0];
        int64_t N = X->dims(0), C = X->dims(1), H = X->dims(2), W = X->dims(3);

        // Default MNIST parameters: 2x2 kernel, stride 2
        int stride = 2, kH = 2, kW = 2;
        int64_t H_out = H / stride;
        int64_t W_out = W / stride;

        std::vector<float> out_data(N * C * H_out * W_out);
        const float* x_ptr = reinterpret_cast<const float*>(X->raw_data().data());

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                float val = x_ptr[n * (C * H * W) + c * (H * W) +
                                                  (oh * stride + kh) * W + (ow * stride + kw)];
                                if (val > max_val)
                                    max_val = val;
                            }
                        }
                        out_data[n * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out + ow] =
                                max_val;
                    }
                }
            }
        }

        auto result = std::make_unique<onnx::TensorProto>();
        for (auto d : {N, C, H_out, W_out})
            result->add_dims(d);
        result->set_raw_data(out_data.data(), out_data.size() * sizeof(float));
        return result;
    }

    TensorPtr log_softmax(const std::vector<const onnx::TensorProto*>& inputs,
                          const onnx::NodeProto& node) {

        assert(inputs.size() == 1);
        const auto* inputTensor = inputs[0];

        // Get axis attribute (default is 1 for LogSoftmax in ONNX)
        int axis = 1;
        for (const auto& attr : node.attribute()) {
            if (attr.name() == "axis")
                axis = attr.i();
        }

        // 1. Initialize the unique_ptr with a copy of the input tensor structure
        TensorPtr outputTensor = std::make_unique<onnx::TensorProto>(*inputTensor);

        // Get dimensions (handling the common 2D [Batch, Classes] case)
        int num_rows = outputTensor->dims(0);
        int num_cols = outputTensor->dims(1);
        float* data = reinterpret_cast<float*>(outputTensor->mutable_raw_data()->data());

        for (int i = 0; i < num_rows; ++i) {
            float* row = data + (i * num_cols);

            // Numerical Stability: Find Max
            float max_val = *std::max_element(row, row + num_cols);

            // Compute Log-Sum-Exp
            float sum_exp = 0.0f;
            for (int j = 0; j < num_cols; ++j) {
                sum_exp += std::exp(row[j] - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);

            // Apply: x - log_sum_exp
            for (int j = 0; j < num_cols; ++j) {
                row[j] = row[j] - log_sum_exp;
            }
        }

        outputTensor->set_name(node.output(0));
        return outputTensor;
    }

}  // namespace tiny_engine::ops