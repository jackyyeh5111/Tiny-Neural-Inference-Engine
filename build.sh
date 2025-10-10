#!/bin/bash

# MNIST ONNX Inference Engine - Build Script
# This script automates the build process for the inference engine

set -e  # Exit on any error

echo "🚀 Building MNIST ONNX Inference Engine..."
echo "============================================"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   (the directory containing CMakeLists.txt)"
    exit 1
fi

# Check for protoc
if ! command -v protoc &> /dev/null; then
    echo "❌ Error: Protocol Buffers compiler (protoc) is not installed"
    echo "   Please install protobuf development libraries"
    exit 1
fi

echo "✅ Dependencies check passed"

# Create build directory
echo "📁 Creating build directory..."
if [ -d "build" ]; then
    echo "   Build directory exists, cleaning..."
    rm -rf build/*
else
    mkdir build
fi

# Configure and build
echo "⚙️  Configuring with CMake..."
cd build
cmake .. || {
    echo "❌ CMake configuration failed"
    exit 1
}

echo "🔨 Building project..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    make -j$(sysctl -n hw.ncpu) || {
        echo "❌ Build failed"
        exit 1
    }
else
    # Linux and others
    make -j$(nproc) || {
        echo "❌ Build failed"
        exit 1
    }
fi

echo "✅ Build completed successfully!"

# Run tests if available
if [ -f "src/gemm_test" ]; then
    echo "🧪 Running tests..."
    make test || {
        echo "⚠️  Some tests failed, but build was successful"
    }
fi

echo ""
echo "🎉 Build complete! You can now run the inference engine:"
echo "   cd build"
echo "   ./src/inference_engine ../models/mnist_ffn.onnx ../inputs/image_0.ubyte"
echo ""
echo "📖 For more information, see README.md"