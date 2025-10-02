#include <iostream>
#include <chrono>
#include "tnie/tnie.h"

using namespace tnie;

void demo_tensor_operations() {
    std::cout << "\n=== Tensor Operations Demo ===" << std::endl;
    
    // Create tensors
    auto tensor1 = Tensor::ones(Shape({2, 3}));
    auto tensor2 = Tensor::random(Shape({2, 3}));
    
    std::cout << "Created tensor1 (ones): shape [2, 3]" << std::endl;
    std::cout << "Created tensor2 (random): shape [2, 3]" << std::endl;
    
    // Print some values
    std::cout << "tensor1[0] = " << tensor1[0] << std::endl;
    std::cout << "tensor2[0] = " << tensor2[0] << std::endl;
    
    // Test tensor operations
    tensor1.fill(2.5f);
    std::cout << "After fill(2.5), tensor1[0] = " << tensor1[0] << std::endl;
}

void demo_gemm_operation() {
    std::cout << "\n=== GEMM Operation Demo ===" << std::endl;
    
    // Create matrices for A * B = C
    // A: 2x3, B: 3x4, C: 2x4
    auto A = Tensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto B = Tensor(Shape({3, 4}), {1.0f, 0.0f, 0.0f, 1.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f, 0.0f});
    
    std::cout << "Matrix A (2x3): [1,2,3; 4,5,6]" << std::endl;
    std::cout << "Matrix B (3x4): identity-like matrix" << std::endl;
    
    // Create GEMM operator
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    
    // Execute GEMM
    std::vector<Tensor> inputs = {A, B};
    std::vector<Tensor> outputs;
    
    auto start = std::chrono::high_resolution_clock::now();
    gemm_op->forward(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "GEMM execution time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Result shape: [" << outputs[0].shape()[0] << ", " << outputs[0].shape()[1] << "]" << std::endl;
    std::cout << "Result[0] = " << outputs[0][0] << std::endl;
    std::cout << "Result[1] = " << outputs[0][1] << std::endl;
}

void demo_graph_execution() {
    std::cout << "\n=== Graph Execution Demo ===" << std::endl;
    
    // Create a simple computation graph:
    // input -> gemm1 -> gemm2 -> output
    
    GraphExecutor executor;
    std::vector<Tensor> tensors;
    
    // Add input tensors
    tensors.push_back(Tensor::ones(Shape({2, 3})));     // tensor 0: input
    tensors.push_back(Tensor::random(Shape({3, 2})));   // tensor 1: weight1
    tensors.push_back(Tensor::random(Shape({2, 1})));   // tensor 2: weight2
    
    std::cout << "Created computation graph with 2 GEMM operations" << std::endl;
    
    // Add operators to graph
    auto gemm1 = OperatorFactory::create_gemm("gemm1");
    auto gemm2 = OperatorFactory::create_gemm("gemm2");
    
    size_t gemm1_node = executor.add_operator(std::move(gemm1), {0, 1}, {3}); // inputs: 0,1 -> output: 3
    size_t gemm2_node = executor.add_operator(std::move(gemm2), {3, 2}, {4}); // inputs: 3,2 -> output: 4
    
    std::cout << "Added GEMM1 node: " << gemm1_node << std::endl;
    std::cout << "Added GEMM2 node: " << gemm2_node << std::endl;
    
    // Execute graph
    auto start = std::chrono::high_resolution_clock::now();
    executor.execute(tensors);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Graph execution time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Final output shape: [" << tensors[4].shape()[0] << ", " << tensors[4].shape()[1] << "]" << std::endl;
    std::cout << "Final result[0] = " << tensors[4][0] << std::endl;
}

void print_system_info() {
    std::cout << "\n=== System Info ===" << std::endl;
    std::cout << "TNIE Version: " << get_version() << std::endl;
    
#ifdef ENABLE_AVX2
    std::cout << "SIMD Support: AVX2 Enabled" << std::endl;
#else
    std::cout << "SIMD Support: Scalar only" << std::endl;
#endif

#ifdef DEBUG
    std::cout << "Build Type: Debug" << std::endl;
#else
    std::cout << "Build Type: Release" << std::endl;
#endif
}

int main() {
    std::cout << "Tiny Neural Inference Engine - Demo Application" << std::endl;
    std::cout << "================================================" << std::endl;
    
    try {
        // Initialize the library
        initialize();
        
        // Run demos
        print_system_info();
        demo_tensor_operations();
        demo_gemm_operation();
        demo_graph_execution();
        
        std::cout << "\n=== All demos completed successfully! ===" << std::endl;
        
        // Cleanup
        finalize();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
