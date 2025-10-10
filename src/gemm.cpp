#include <assert.h>

#include "gemm.h"

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
void gemm(const float *A, const float *B, const float *C, float *out, const int n, const int m, const int k)
{
    // Perform matrix multiplication: out = A * B
    for (int r = 0; r < n; ++r)      // Iterate through rows of A
    {
        for (int c = 0; c < k; ++c)  // Iterate through columns of B
        {
            float res = 0.0f;
            // Compute dot product of row r of A with column c of B
            for (int i = 0; i < m; ++i)
            {
                res += A[r * m + i] * B[i * k + c];
            }
            out[r * k + c] = res;
        }
    }

    // Add bias term: out = out + C
    for (int r = 0; r < n; ++r)
    {
        for (int c = 0; c < k; ++c)
        {
            out[r * k + c] += C[r * k + c];
        }
    }
}