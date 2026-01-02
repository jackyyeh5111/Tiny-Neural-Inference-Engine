#include "utils.h"

namespace tiny_engine::utils {

    std::vector<float> softmax(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        float max_val = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;

        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - max_val);  // Numerical stability
            sum += output[i];
        }
        for (float& val : output)
            val /= sum;
        return output;
    }

    int argmax(const std::vector<float>& input) {
        return std::distance(input.begin(), std::max_element(input.begin(), input.end()));
    }

    TensorPtr read_input(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open input file: " + filename);
        }

        // MNIST images are 28x28 = 784 pixels
        const size_t num_pixels = 784;
        std::vector<unsigned char> bytes(num_pixels);
        file.read(reinterpret_cast<char*>(bytes.data()), num_pixels);

        if (file.gcount() != num_pixels) {
            throw std::runtime_error("Input file size mismatch. Expected 784 bytes.");
        }

        // Convert to float and normalize (0-255 -> 0.0-1.0)
        std::vector<float> floatValues(num_pixels);
        for (size_t i = 0; i < num_pixels; ++i) {
            floatValues[i] = static_cast<float>(bytes[i]) / 255.0f;
        }

        // Create the Tensor using unique_ptr
        auto modelInput = std::make_unique<onnx::TensorProto>();

        // This name should match the first input name of your graph
        modelInput->set_name("onnx::Flatten_0");
        modelInput->set_data_type(onnx::TensorProto::FLOAT);

        // Set dimensions: Batch=1, Channels=1, Height=28, Width=28
        modelInput->add_dims(1);
        modelInput->add_dims(1);
        modelInput->add_dims(28);
        modelInput->add_dims(28);

        // Copy data into the Protobuf internal buffer
        modelInput->set_raw_data(floatValues.data(), floatValues.size() * sizeof(float));

        std::cout << "[LOADER] Input loaded and normalized. Shape: [1, 1, 28, 28]" << std::endl;

        return modelInput;
    }

}  // namespace tiny_engine::utils