#include <iostream>
#include <chrono>
#include <vector>
#include "tnie/tnie.h"

using namespace tnie;

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

void benchmark_gemm(size_t M, size_t N, size_t K, int iterations = 100) {
    std::cout << "\n=== GEMM Benchmark: " << M << "x" << K << " * " << K << "x" << N << " ===" << std::endl;
    
    // Create input matrices
    auto A = Tensor::random(Shape({M, K}));
    auto B = Tensor::random(Shape({K, N}));
    
    auto gemm_op = OperatorFactory::create_gemm("benchmark_gemm");
    
    std::vector<Tensor> inputs = {A, B};
    std::vector<Tensor> outputs;
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        outputs.clear();
        gemm_op->forward(inputs, outputs);
    }
    
    // Benchmark
    Timer timer;
    timer.start();
    
    for (int i = 0; i < iterations; ++i) {
        outputs.clear();
        gemm_op->forward(inputs, outputs);
    }
    
    double total_time = timer.stop();
    double avg_time = total_time / iterations;
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K; // Each output element requires K multiplications and K-1 additions
    double gflops = (flops * iterations) / (total_time * 1e3); // Convert microseconds to seconds and scale to GFLOPS
    
    std::cout << "Average time per iteration: " << avg_time << " microseconds" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
#ifdef ENABLE_AVX2
    std::cout << "SIMD: AVX2 enabled" << std::endl;
#else
    std::cout << "SIMD: Scalar only" << std::endl;
#endif
}

void benchmark_tensor_operations() {
    std::cout << "\n=== Tensor Operations Benchmark ===" << std::endl;
    
    const size_t size = 1000000; // 1M elements
    const int iterations = 100;
    
    Timer timer;
    
    // Tensor creation
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        auto tensor = Tensor::zeros(Shape({size}));
    }
    double creation_time = timer.stop() / iterations;
    
    // Fill operation
    auto tensor = Tensor::zeros(Shape({size}));
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        tensor.fill(1.5f);
    }
    double fill_time = timer.stop() / iterations;
    
    // Element access
    timer.start();
    float sum = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        for (size_t j = 0; j < std::min(size, 1000UL); ++j) {
            sum += tensor[j];
        }
    }
    double access_time = timer.stop() / iterations;
    
    std::cout << "Tensor creation (" << size << " elements): " << creation_time << " microseconds" << std::endl;
    std::cout << "Fill operation: " << fill_time << " microseconds" << std::endl;
    std::cout << "Element access (1000 elements): " << access_time << " microseconds" << std::endl;
    std::cout << "Sum: " << sum << " (to prevent optimization)" << std::endl;
}

void benchmark_graph_execution() {
    std::cout << "\n=== Graph Execution Benchmark ===" << std::endl;
    
    const int iterations = 100;
    Timer timer;
    
    // Create a simple 3-layer network graph
    GraphExecutor executor;
    
    auto gemm1 = OperatorFactory::create_gemm("layer1");
    auto gemm2 = OperatorFactory::create_gemm("layer2");
    auto gemm3 = OperatorFactory::create_gemm("layer3");
    
    executor.add_operator(std::move(gemm1), {0, 1}, {3}); // input * weight1 -> hidden1
    executor.add_operator(std::move(gemm2), {3, 2}, {4}); // hidden1 * weight2 -> hidden2
    executor.add_operator(std::move(gemm3), {4, 5}, {6}); // hidden2 * weight3 -> output
    
    // Prepare tensors
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::random(Shape({32, 128}))); // batch_size=32, input_dim=128
    tensors.push_back(Tensor::random(Shape({128, 64}))); // weight1: 128->64
    tensors.push_back(Tensor::random(Shape({64, 32})));  // weight2: 64->32
    tensors.push_back(Tensor());                         // hidden1 (will be created)
    tensors.push_back(Tensor());                         // hidden2 (will be created)
    tensors.push_back(Tensor::random(Shape({32, 10})));  // weight3: 32->10
    tensors.push_back(Tensor());                         // output (will be created)
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        auto tensors_copy = tensors;
        executor.execute(tensors_copy);
    }
    
    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        auto tensors_copy = tensors;
        executor.execute(tensors_copy);
    }
    double total_time = timer.stop();
    double avg_time = total_time / iterations;
    
    std::cout << "3-layer network (32x128->64->32->10): " << avg_time << " microseconds per forward pass" << std::endl;
    std::cout << "Throughput: " << (1e6 / avg_time) << " inferences per second" << std::endl;
}

int main() {
    std::cout << "Tiny Neural Inference Engine - Benchmarks" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    tnie::initialize();
    
    try {
        // Print system info
        std::cout << "Version: " << get_version() << std::endl;
#ifdef ENABLE_AVX2
        std::cout << "SIMD: AVX2 enabled" << std::endl;
#else
        std::cout << "SIMD: Scalar only" << std::endl;
#endif

        // Run benchmarks
        benchmark_tensor_operations();
        
        // GEMM benchmarks with different sizes
        benchmark_gemm(64, 64, 64);     // Small
        benchmark_gemm(256, 256, 256);  // Medium
        benchmark_gemm(512, 512, 512);  // Large
        
        benchmark_graph_execution();
        
        std::cout << "\n=== Benchmarks completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
    
    tnie::finalize();
    return 0;
}
