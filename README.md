# Tiny Neural Inference Engine (TNIE)

A minimal neural network inference engine with SIMD optimizations and multi-threading support, built for educational purposes and performance experimentation.

## Project Goals

- **Minimal tensor and operator abstraction**: Clean, efficient tensor operations with shape management
- **SIMD optimizations**: AVX2 implementations for GEMM and Conv2D with scalar fallbacks
- **Graph execution**: Dependency-aware operator scheduling and execution
- **Multi-threading**: Worker thread pool with work stealing for parallel operator execution
- **Performance profiling**: Comprehensive benchmarking infrastructure for SIMD vs scalar and single vs multi-thread comparisons

## Features

- ✅ **Tensor operations**: N-dimensional tensors with efficient memory management
- ✅ **Core operators**: GEMM (matrix multiplication) and Conv2D implementations
- ✅ **Graph execution**: Topological sorting and dependency-aware execution
- ✅ **SIMD support**: AVX2 optimizations with runtime detection
- ✅ **Testing**: Comprehensive unit tests with GoogleTest
- ✅ **Benchmarking**: Performance measurement infrastructure
- 🔄 **Multi-threading**: (Coming in future weeks)
- 🔄 **Advanced optimizations**: (Coming in future weeks)

## Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16 or later
- AVX2 support (optional, falls back to scalar)

### Building

```bash
# Clone the repository
git clone https://github.com/jackyyeh5111/Tiny-Neural-Inference-Engine.git
cd Tiny-Neural-Inference-Engine

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j$(nproc)
```

### Build Options

```bash
# Enable/disable features
cmake .. -DBUILD_TESTS=ON          # Build unit tests (default: ON)
cmake .. -DBUILD_BENCHMARKS=ON     # Build benchmarks (default: ON)
cmake .. -DENABLE_SIMD=ON          # Enable AVX2 optimizations (default: ON)

# Build types
cmake .. -DCMAKE_BUILD_TYPE=Release  # Optimized build
cmake .. -DCMAKE_BUILD_TYPE=Debug    # Debug build with symbols
```

### Running

```bash
# Run the demo application
./src/tnie_main

# Run unit tests
./tests/tnie_tests

# Run benchmarks
./benchmarks/tnie_benchmark
```

## Architecture Overview

### Core Components

1. **Tensor (`tnie/tensor.h`)**: N-dimensional array abstraction with efficient memory management
2. **Operators (`tnie/operator.h`)**: Base classes for GEMM, Conv2D, and other operations
3. **Graph Executor (`tnie/graph.h`)**: Dependency-aware execution engine
4. **TNIE (`tnie/tnie.h`)**: Main library interface

### Example Usage

```cpp
#include "tnie/tnie.h"
using namespace tnie;

// Initialize library
tnie::initialize();

// Create tensors
auto A = Tensor::random(Shape({128, 256}));
auto B = Tensor::random(Shape({256, 64}));

// Create and execute GEMM operation
auto gemm_op = OperatorFactory::create_gemm("matmul");
std::vector<Tensor> inputs = {A, B};
std::vector<Tensor> outputs;
gemm_op->forward(inputs, outputs);

// Use graph executor for complex workflows
GraphExecutor executor;
executor.add_operator(std::move(gemm_op), {0, 1}, {2});

std::vector<Tensor> tensors = {A, B};
executor.execute(tensors);

// Cleanup
tnie::finalize();
```

## Performance

Current implementation includes:

- **SIMD optimizations**: AVX2 implementations for GEMM operations
- **Memory alignment**: 32-byte aligned memory for optimal SIMD performance
- **Efficient graph execution**: Topological sorting minimizes redundant operations

### Benchmark Results

Run `./benchmarks/tnie_benchmark` to see performance on your system. Example output:

```
=== GEMM Benchmark: 256x256 * 256x256 ===
Average time per iteration: 1250 microseconds
Performance: 26.8 GFLOPS
SIMD: AVX2 enabled
```

## Development

### Project Structure

```
├── CMakeLists.txt          # Root CMake configuration
├── README.md               # This file
├── include/tnie/           # Public headers
│   ├── tensor.h           # Tensor abstraction
│   ├── operator.h         # Operator base classes
│   ├── graph.h            # Graph executor
│   └── tnie.h             # Main library header
├── src/                   # Implementation files
│   ├── CMakeLists.txt     # Source CMake config
│   ├── main.cpp           # Demo application
│   ├── tensor.cpp         # Tensor implementation
│   ├── operator.cpp       # Operator implementations
│   ├── graph.cpp          # Graph executor implementation
│   └── tnie.cpp           # Library utilities
├── tests/                 # Unit tests
│   ├── CMakeLists.txt     # Test CMake config
│   ├── test_tensor.cpp    # Tensor tests
│   ├── test_operator.cpp  # Operator tests
│   └── test_graph.cpp     # Graph execution tests
└── benchmarks/            # Performance benchmarks
    ├── CMakeLists.txt     # Benchmark CMake config
    └── benchmark_main.cpp # Benchmark suite
```

### Running Tests

```bash
# Run all tests
make test

# Run tests with verbose output
ctest --verbose

# Run specific test
./tests/tnie_tests --gtest_filter="TensorTest.*"
```

### Contributing

1. Follow the existing code style (C++17, 4-space indentation)
2. Add unit tests for new features
3. Update benchmarks for performance-critical changes
4. Ensure all tests pass before submitting

## Roadmap

### Week 0 ✅ (Current)
- [x] CMake project setup
- [x] Basic tensor and operator abstractions
- [x] GEMM and Conv2D implementations
- [x] Graph executor
- [x] Unit testing infrastructure
- [x] Basic benchmarking

### Future Weeks
- **Week 1**: Multi-threading with worker pools
- **Week 2**: Advanced SIMD optimizations (kernel optimization, better memory access patterns)
- **Week 3**: Additional operators (ReLU, Softmax, BatchNorm)
- **Week 4**: Memory pool and optimization passes
- **Week 5**: Model loading and real-world benchmarks

## License

This project is for educational purposes. See LICENSE file for details.

## Acknowledgments

Built as a learning exercise in high-performance computing, SIMD optimization, and neural network inference.
