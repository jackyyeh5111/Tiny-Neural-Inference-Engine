#include "ops.h"

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
        std::vector<float> b_processed;
        if (transB) {
            b_processed.resize(K * N);
            int orig_rows = B->dims(0);  // K
            int orig_cols = B->dims(1);  // N
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
                    c_data[i * N + j] = c_ptr[c_idx] * beta;
                }
            }
        }

        // 5. Execute Math
        std::vector<float> out_data(M * N);
        // Call your raw gemm function from gemm.h
        _gemm(a_ptr, b_ptr, c_data.data(), out_data.data(), M, K, N);

        // 6. Wrap in TensorProto
        auto result = std::make_unique<onnx::TensorProto>();
        result->set_name(node.output(0));
        result->set_data_type(onnx::TensorProto::FLOAT);
        result->add_dims(M);
        result->add_dims(N);
        result->set_raw_data(out_data.data(), out_data.size() * sizeof(float));

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
               const int n,
               const int m,
               const int k) {
        // Perform matrix multiplication: out = A * B
        for (int r = 0; r < n; ++r)  // Iterate through rows of A
        {
            for (int c = 0; c < k; ++c)  // Iterate through columns of B
            {
                float res = 0.0f;
                // Compute dot product of row r of A with column c of B
                for (int i = 0; i < m; ++i) {
                    res += A[r * m + i] * B[i * k + c];
                }
                out[r * k + c] = res;
            }
        }

        // Add bias term: out = out + C
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; ++c) {
                out[r * k + c] += C[r * k + c];
            }
        }
    }

}  // namespace tiny_engine::ops