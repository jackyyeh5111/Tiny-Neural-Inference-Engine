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

    TensorPtr read_mnist_input(const std::string& filename) {
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

        // Convert to float, Scale (0-1), and Normalize (Standardize)
        std::vector<float> floatValues(num_pixels);
        for (size_t i = 0; i < num_pixels; ++i) {
            floatValues[i] = static_cast<float>(bytes[i]);
        }

        auto modelInput = std::make_unique<onnx::TensorProto>();

        modelInput->set_name("dummy_input");  // input name will be modified later
        modelInput->set_data_type(onnx::TensorProto::FLOAT);

        modelInput->add_dims(1);
        modelInput->add_dims(1);
        modelInput->add_dims(28);
        modelInput->add_dims(28);

        modelInput->set_raw_data(floatValues.data(), floatValues.size() * sizeof(float));

        return modelInput;
    }

    void print_tensor_dims(const std::string& name, const onnx::TensorProto& tensor) {
        std::cout << name << " shape: [";
        for (int i = 0; i < tensor.dims_size(); ++i) {
            std::cout << tensor.dims(i) << (i == tensor.dims_size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
}  // namespace tiny_engine::utils