# Quick Start Guide

## TL;DR - Get Running in 2 Minutes

1. **Install dependencies (if not already installed):**
   ```bash
   # macOS
   brew install cmake protobuf
   
   # Ubuntu/Debian
   sudo apt install build-essential cmake libprotobuf-dev protobuf-compiler
   ```

2. **Build the project:**
   ```bash
   ./build.sh
   ```

3. **Run MNIST inference:**
   ```bash
   cd build
   ./src/inference_engine ../models/mnist_ffn.onnx ../inputs/image_0.ubyte
   ```

4. **Expected output:**
   ```
   === MNIST CLASSIFICATION RESULTS ===
   Predicted digit: [0-9]
   Confidence: XX.XX%
   [Detailed probability distribution]
   ```

## Generate More Test Images

```bash
cd utils
python3 create_test_image.py all  # Creates test images for digits 0-9
cd ../build
./src/inference_engine ../models/mnist_ffn.onnx ../inputs/digit_5_test.ubyte
```

## Troubleshooting

- **Build fails:** Check that cmake and protobuf are installed
- **Can't find model:** Make sure you're running from the build/ directory
- **Wrong predictions:** The test images are synthetic - real MNIST data will work better

For detailed instructions, see the main [README.md](README.md).