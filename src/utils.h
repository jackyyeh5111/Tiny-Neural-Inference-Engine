#pragma once

#include "config.h"

namespace tiny_engine::utils {

    std::vector<float> softmax(const std::vector<float>& input);

    int argmax(const std::vector<float>& input);

    TensorPtr read_input(const std::string& filename);

    void print_tensor_dims(const std::string& name, const onnx::TensorProto& tensor);

}  // namespace tiny_engine::utils