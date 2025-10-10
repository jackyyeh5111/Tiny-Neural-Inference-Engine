# MNIST ONNX Inference Engine

A lightweight C++ inference engine for MNIST digit classification using ONNX models. This engine loads ONNX neural network models and performs inference on 28x28 grayscale images to classify handwritten digits (0-9).

## Features

- ✅ ONNX model loading and parsing using Protocol Buffers
- ✅ Support for common neural network operations:
  - Flatten (tensor reshaping)
  - GEMM (General Matrix Multiplication) - fully connected layers
  - ReLU activation function
  - Constant value handling
- ✅ MNIST image loading from binary files
- ✅ Softmax classification with confidence scores
- ✅ Detailed output visualization

## Prerequisites

### System Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- CMake 3.10 or higher
- Protocol Buffers library and compiler

### Installing Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake libprotobuf-dev protobuf-compiler
```

#### macOS:
```bash
# Using Homebrew
brew install cmake protobuf

# Using MacPorts
sudo port install cmake protobuf3-cpp
```

#### Optional: For running tests
```bash
# Ubuntu/Debian
sudo apt install libgtest-dev

# macOS
brew install googletest
```

## Building the Project

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/inference_engine
   ```

2. **Create build directory:**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake:**
   ```bash
   cmake ..
   ```

4. **Build the project:**
   ```bash
   make -j$(nproc)  # Linux
   make -j$(sysctl -n hw.ncpu)  # macOS
   ```

5. **Run tests (optional):**
   ```bash
   make test
   # Or run individual test
   ./src/gemm_test
   ```

## Usage

### Basic Usage
```bash
./src/inference_engine <model.onnx> <input_image.ubyte>
```

### Example Commands
```bash
# Using existing test data
./src/inference_engine ../models/mnist_ffn.onnx ../inputs/image_0.ubyte

# Using generated test images (after running the generator)
./src/inference_engine ../models/mnist_ffn.onnx ../inputs/digit_5_test.ubyte
```

## Input Image Format

The engine expects input images in a specific binary format:
- **File size:** Exactly 784 bytes
- **Format:** Raw binary data (unsigned 8-bit integers)
- **Dimensions:** 28x28 pixels, flattened row-by-row
- **Pixel values:** 0-255 (0 = black, 255 = white)

### Creating Test Images

Use the provided Python utility to generate test images:

```bash
cd utils

# Generate all digits (0-9)
python3 create_test_image.py all

# Generate specific digit
python3 create_test_image.py 5

# Generate random test image
python3 create_test_image.py random
```

### Converting Real MNIST Data

If you have MNIST dataset files, you can extract individual images:

```python
import struct
import numpy as np

def extract_mnist_image(idx_file, index, output_file):
    with open(idx_file, 'rb') as f:
        # Skip header (16 bytes)
        f.seek(16 + index * 784)
        image_data = f.read(784)
        
    with open(output_file, 'wb') as f:
        f.write(image_data)

# Example usage
extract_mnist_image('t10k-images.idx3-ubyte', 0, 'test_image.ubyte')
```

## Expected Output

When you run the inference engine, you'll see:

1. **Model Information:**
   - Input/output tensor names
   - Network architecture (nodes and operations)
   - Weight loading status

2. **Processing Steps:**
   - Each operation execution with intermediate results
   - Tensor dimensions and data flow

3. **Classification Results:**
   ```
   === MNIST CLASSIFICATION RESULTS ===
   Output tensor dimensions: 1 x 10 (total elements: 10)
   
   Raw model output (logits): -2.1234, 0.5678, 3.1415, ...
   
   --- CLASSIFICATION RESULTS ---
   Predicted digit: 5
   Confidence: 89.23%
   
   Probability distribution:
     Digit 0: 0.0012 (0.1%)
     Digit 1: 0.0034 (0.3%)
     ...
     Digit 5: 0.8923 (89.2%)
     ...
   
   Confidence bar for digit 5: ████████████████████ 89.2%
   ```

## ONNX Model Requirements

The engine currently supports ONNX models with these operations:
- **Flatten:** Reshape tensors (typically from 1x1x28x28 to 1x784)
- **Gemm:** Matrix multiplication with bias (fully connected layers)
- **Relu:** ReLU activation function
- **Constant:** Constant value nodes

### Typical MNIST Model Architecture:
```
Input (1x1x28x28) 
    ↓
Flatten → (1x784)
    ↓
Gemm → (1x128) + ReLU
    ↓
Gemm → (1x64) + ReLU  
    ↓
Gemm → (1x10)
    ↓
Output (1x10) → Softmax → Classification
```

## Troubleshooting

### Common Issues:

1. **"Failed to open the ONNX model file"**
   - Check file path and permissions
   - Ensure the .onnx file is valid

2. **"Failed to parse the ONNX model"**
   - Model might be corrupted or unsupported version
   - Check ONNX model version compatibility

3. **"Input: X not in weights"**
   - Model has missing initializers
   - Check if all required weights are present

4. **"Matrix dimensions are not compatible"**
   - Input image size doesn't match model expectations
   - Ensure input is exactly 784 bytes (28x28)

5. **Build errors with protobuf:**
   ```bash
   # Clear build directory and rebuild
   rm -rf build
   mkdir build && cd build
   cmake .. && make
   ```

### Debug Mode:
To get more detailed output, build in debug mode:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## File Structure
```
inference_engine/
├── CMakeLists.txt          # Main build configuration
├── README.md               # This file
├── models/
│   └── mnist_ffn.onnx     # Pre-trained MNIST model
├── inputs/
│   ├── image_0.ubyte      # Sample test image
│   └── ...                # Additional test images
├── src/
│   ├── CMakeLists.txt     # Source build configuration
│   ├── main.cpp           # Main inference engine
│   ├── gemm.h/.cpp        # Matrix multiplication implementation
│   ├── onnx-ml.proto      # ONNX protobuf definition
│   └── test/
│       └── gemm_test.cpp  # Unit tests
└── utils/
    └── create_test_image.py # Test image generator
```

## Development

### Adding New Operations:
1. Add function declaration in `main.cpp`
2. Implement the operation following existing patterns
3. Add to the operation dispatch in the main loop
4. Test with appropriate ONNX models

### Contributing:
- Follow existing code style and documentation patterns
- Add unit tests for new functionality
- Update this README for any new features or requirements

## License

This project is provided as educational material for understanding ONNX inference engines and neural network inference.