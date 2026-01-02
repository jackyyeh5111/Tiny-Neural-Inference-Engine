#pragma once

#include "config.h"

namespace tiny_engine::utils {

    std::vector<float> softmax(const std::vector<float>& input);

    int argmax(const std::vector<float>& input);

    TensorPtr read_input(const std::string& filename);

}  // namespace tiny_engine::utils