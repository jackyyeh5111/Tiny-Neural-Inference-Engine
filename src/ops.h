#pragma once

#include "config.h"

namespace tiny_engine::ops {

    TensorPtr relu(const std::vector<const onnx::TensorProto*>& inputs,
                   const onnx::NodeProto& node);

    TensorPtr flatten(const std::vector<const onnx::TensorProto*>& inputs,
                      const onnx::NodeProto& node);

    TensorPtr gemm(const std::vector<const onnx::TensorProto*>& inputs,
                   const onnx::NodeProto& node);

    TensorPtr conv(const std::vector<const onnx::TensorProto*>& inputs,
                   const onnx::NodeProto& node);

    TensorPtr maxpool(const std::vector<const onnx::TensorProto*>& inputs,
                      const onnx::NodeProto& node);

    TensorPtr log_softmax(const std::vector<const onnx::TensorProto*>& inputs,
                          const onnx::NodeProto& node);

    void _gemm(const float* A, const float* B, const float* C, float* out, int m, int n, int k);
}
