#include "InferEnginer.h"
#include "ops.h"
#include "utils.h"

using namespace tiny_engine;

InferEngine::InferEngine() {
    _register_operators();
}

bool InferEngine::load_model(const std::string& filename) {
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    if (!model_.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse ONNX model." << std::endl;
        return false;
    }

    // Load initializers (weights) into the memory map
    for (const auto& init : model_.graph().initializer()) {
        weights_map_[init.name()] = &init;
    }
    return true;
}

void InferEngine::set_input(TensorPtr input_tensor) {
    input_storage_ = std::move(input_tensor);

    // set input name
    const auto& input_name = model_.graph().node()[0].input(0);
    input_storage_->set_name(input_name);

    std::cout << "Input tensor name: " << input_storage_->name() << std::endl;
    weights_map_[input_storage_->name()] = input_storage_.get();
}

void InferEngine::run() {
    const auto& graph = model_.graph();

    // print all name in weights_map_
    std::cout << "\n=== Weights Map Contents ===" << std::endl;
    for (const auto& pair : weights_map_) {
        std::cout << "Tensor Name: " << pair.first << std::endl;
    }
    std::cout << "============================\n" << std::endl;

    // TODO: Topological sort of nodes if necessary
    // Execute each node in the graph
    for (const auto& node : graph.node()) {
        std::cout << "[EXECUTING] " << node.op_type() << ": " << node.name() << std::endl;

        // Resolve Inputs
        std::vector<const onnx::TensorProto*> inputs;
        for (const auto& in_name : node.input()) {
            std::cout << "  - Input: " << in_name << std::endl;

            if (weights_map_.count(in_name) == 0) {
                throw std::runtime_error("Input tensor " + in_name + " not found in weights_map_");
            }

            inputs.push_back(weights_map_[in_name]);
        }

        // Execute Operator
        if (op_registry_.count(node.op_type())) {
            TensorPtr output = op_registry_[node.op_type()](inputs, node);

            // Store output in intermediate map
            std::string out_name = node.output(0);
            weights_map_[out_name] = output.get();
            intermediate_tensors_[out_name] = std::move(output);
        } else {
            std::cerr << "Warning: Unsupported operator " << node.op_type() << std::endl;
        }
    }

    // Automatically find and process the graph's final output
    if (graph.output_size() > 0) {
        std::string final_output_name = graph.output(0).name();
        _process_results(final_output_name);
    }
}

const onnx::TensorProto* InferEngine::get_output(const std::string& name) {
    return weights_map_.count(name) ? weights_map_[name] : nullptr;
}

void InferEngine::_process_results(const std::string& output_name) {
    const onnx::TensorProto* res = get_output(output_name);
    if (!res) {
        std::cerr << "Error: Output tensor " << output_name << " not found." << std::endl;
        return;
    }

    // Extract data
    const float* raw_data = reinterpret_cast<const float*>(res->raw_data().data());
    size_t count = res->raw_data().size() / sizeof(float);
    std::vector<float> logits(raw_data, raw_data + count);

    // Run MNIST-specific classification
    _display_mnist_classification(logits);
}

void InferEngine::_display_mnist_classification(const std::vector<float>& logits) {
    auto probs = utils::softmax(logits);
    int prediction = utils::argmax(probs);
    float confidence = probs[prediction];

    std::cout << "\n================================" << std::endl;
    std::cout << "   MNIST CLASSIFICATION RESULT  " << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "PREDICTION : " << prediction << std::endl;
    std::cout << "CONFIDENCE : " << std::fixed << std::setprecision(2) << (confidence * 100.0f)
              << "%" << std::endl;

    // Print probability bar chart
    std::cout << "\nDistribution:" << std::endl;
    for (int i = 0; i < probs.size(); ++i) {
        int bar_width = static_cast<int>(probs[i] * 30);
        std::cout << i << ": [" << std::string(bar_width, 'X') << std::string(30 - bar_width, ' ')
                  << "] " << std::setprecision(4) << probs[i] << std::endl;
    }
}

void InferEngine::_register_operators() {
    op_registry_["Relu"] = ops::relu;
    op_registry_["Flatten"] = ops::flatten;
    op_registry_["Gemm"] = ops::gemm;
    op_registry_["Conv"] = ops::conv;
    op_registry_["MaxPool"] = ops::maxpool;
    op_registry_["LogSoftmax"] = ops::log_softmax;
}
