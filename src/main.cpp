#include <iostream>
#include <fstream>
#include <random>
#include <assert.h>
#include <vector>
#include <iomanip> // For std::hex
#include <numeric> // for std::accumulate
#include <span>
#include <cmath>   // For exp() function in softmax
#include <algorithm> // For max_element in argmax

#include "onnx-ml.pb.h" // Include the generated header
#include "gemm.h"

onnx::TensorProto *relu(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node);
onnx::TensorProto *flatten(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node);
onnx::TensorProto *read_input(std::string filename);
onnx::TensorProto* gemm(const std::vector<const onnx::TensorProto*>& inputs, const onnx::NodeProto& node);
float extract_const(onnx::NodeProto node);
void printRawTensorData(const onnx::TensorProto &tensor);
int getFlattenAxis(const onnx::NodeProto &node);

// Classification helper functions for MNIST digit recognition
std::vector<float> softmax(const std::vector<float>& logits);
int argmax(const std::vector<float>& probabilities);
void classify_mnist_output(const float* output_data, int output_size);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.data>" << std::endl;
        return 1;
    }

    std::string modelFile = argv[1];
    std::string inputFile = argv[2];

    std::cout << "ONNX model file: " << modelFile << std::endl;
    std::cout << "User input file: " << inputFile << std::endl;

    std::fstream input(modelFile, std::ios::in | std::ios::binary);
    if (!input.is_open())
    {
        std::cerr << "Failed to open the ONNX model file!" << std::endl;
        return -1;
    }

    onnx::ModelProto model;

    // Parse the model from the file
    if (!model.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the ONNX model!" << std::endl;
        return -1;
    }

    const onnx::GraphProto &graph = model.graph();

    // Example: Print the model's graph input names
    std::cout << "Model Inputs:" << std::endl;
    for (const auto &input : graph.input())
    {
        std::cout << "  - " << input.name() << std::endl;
    }

    // Print graph outputs
    std::cout << "\nOutputs:\n";
    for (const auto &output : graph.output())
    {
        std::cout << "  - Name: " << output.name() << ", Type: " << output.type().denotation() << std::endl;
    }
    assert(graph.output_size() == 1);
    const std::string graph_output = graph.output()[0].name();

    // Print nodes (operators)
    std::cout << "\nNodes (Operators):\n";
    for (const auto &node : graph.node())
    {
        std::cout << "  - OpType: " << node.op_type();
        std::cout << ", Name: " << node.name();
        std::cout << ", Inputs: [";
        for (const auto &input : node.input())
        {
            std::cout << input << " ";
        }
        std::cout << "], Outputs: [";
        for (const auto &output : node.output())
        {
            std::cout << output << " ";
        }
        std::cout << "]\n";
    }

    std::map<std::string, const onnx::TensorProto *> weights;
    for (const auto &init : graph.initializer())
    {
        std::cout << "Adding initializer: " << init.name() << std::endl;
        weights[init.name()] = &init;
    }

    // Make mock input.
    onnx::TensorProto *modelInput = read_input(inputFile);
    weights[modelInput->name()] = modelInput;

    // Iterate over nodes (topologically sorted)
    std::cout << "iterating over graph" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << std::endl;
    for (const auto &node : graph.node())
    {

        std::cout << "Node: " << node.name() << std::endl;

        std::string op_type = node.op_type();
        std::vector<const onnx::TensorProto *> inputs;

        std::cout << "Inputs: ";
        for (const std::string input_name : node.input())
        {

            if (weights.find(input_name) == weights.end())
            {
                std::cerr << "Input: " << input_name << " not in weights. Aborting." << std::endl;
                exit(1);
            }
            const onnx::TensorProto *in = weights[input_name];
            std::cout << in->name() << ", ";
            inputs.push_back(in);
        }
        std::cout << std::endl;

        if (op_type == "Gemm")
        {
            onnx::TensorProto* result = gemm(inputs, node);
            weights[result->name()] = result;
        }
        else if (op_type == "Constant")
        {
            float constant = extract_const(node);
        }
        else if (op_type == "Flatten")
        {
            onnx::TensorProto* output = flatten(inputs, node);
            weights[output->name()] = output;
        }
        else if (op_type == "Relu")
        {
            onnx::TensorProto* output = relu(inputs, node);
            weights[output->name()] = output;
        }
        else
        {
            std::cerr << "Unsupported operation: " << op_type << ". Skipping..." << std::endl;
        }
        std::cout << std::endl;
    }
    // =================================================================
    // SECTION: Final Output Processing and MNIST Classification
    // =================================================================
    
    if (weights.find(graph_output) != weights.end())
    {
        std::cout << "\n=== MNIST CLASSIFICATION RESULTS ===" << std::endl;
        const onnx::TensorProto *res = weights[graph_output];
        const float *output_data = (const float *)res->raw_data().data();
        
        // Determine output size based on tensor dimensions
        int output_size = 1;
        std::cout << "Output tensor dimensions: ";
        for (int i = 0; i < res->dims_size(); ++i) {
            std::cout << res->dims(i);
            if (i < res->dims_size() - 1) std::cout << " x ";
            output_size *= res->dims(i);
        }
        std::cout << " (total elements: " << output_size << ")" << std::endl;
        
        // Print raw logits
        std::cout << "\nRaw model output (logits): ";
        for (int i = 0; i < output_size; ++i)
        {
            std::cout << std::fixed << std::setprecision(4) << output_data[i];
            if (i < output_size - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Apply classification logic
        classify_mnist_output(output_data, output_size);
    }
    else
    {
        std::cerr << "Error: Could not find model output tensor '" << graph_output << "'" << std::endl;
    }

    return 0;
}

onnx::TensorProto *read_input(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::vector<unsigned char> bytes(784); // Preallocate for 784 bytes
    file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());

    std::vector<float> floatValues(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        floatValues[i] = static_cast<float>(bytes[i]); // Direct conversion and normalize.
    }

    std::cout << "File size: " << floatValues.size() << std::endl;
    assert(floatValues.size() == 784);

    onnx::TensorProto *modelInput = new onnx::TensorProto;
    modelInput->set_name("onnx::Flatten_0");

    // Set data type to FLOAT
    modelInput->set_data_type(onnx::TensorProto::FLOAT);

    // Set dimensions to [1, 1, 28, 28]
    modelInput->add_dims(1);
    modelInput->add_dims(1);
    modelInput->add_dims(28);
    modelInput->add_dims(28);

    // Fill with 1s (assuming FLOAT data type)
    modelInput->set_raw_data(floatValues.data(), floatValues.size() * sizeof(float));

    int float_size = modelInput->raw_data().size() / sizeof(float);
    std::cout << "input float_size: " << float_size << std::endl;
    std::cout << "input:";
    const float *raw = (float *)modelInput->raw_data().data();
    for (int i = 0; i < float_size; ++i)
    {
        std::cout << " " << raw[i];
    }
    std::cout << std::endl;

    return modelInput;
}

onnx::TensorProto* gemm(const std::vector<const onnx::TensorProto*>& inputs, const onnx::NodeProto& node) {
    std::cout << "Op: Gemm" << std::endl;

    if (inputs.size() != 3) {
        throw std::invalid_argument("Gemm operator expects exactly three input tensors.");
    }

    // Parse GEMM attributes
    float alpha = 1.0f, beta = 1.0f;
    int transA = 0, transB = 0;
    
    for (const auto& attr : node.attribute()) {
        if (attr.name() == "alpha") {
            alpha = attr.f();
        } else if (attr.name() == "beta") {
            beta = attr.f();
        } else if (attr.name() == "transA") {
            transA = attr.i();
        } else if (attr.name() == "transB") {
            transB = attr.i();
        }
    }
    
    std::cout << "GEMM attributes: alpha=" << alpha << ", beta=" << beta 
              << ", transA=" << transA << ", transB=" << transB << std::endl;

    const auto* A = inputs[0];
    const auto* B = inputs[1];
    const auto* C = inputs[2];

    // Input Validation - A should be 2D, B should be 2D, C should be 1D (bias vector)
    if (A->dims_size() != 2 || B->dims_size() != 2 || C->dims_size() != 1) {
        std::cerr << "A dims: " << A->dims_size() << " B dims: " << B->dims_size() << " C dims: " << C->dims_size() << std::endl;
        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    
    // Get effective dimensions after potential transpose
    int64_t A_rows = transA ? A->dims(1) : A->dims(0);
    int64_t A_cols = transA ? A->dims(0) : A->dims(1);
    int64_t B_rows = transB ? B->dims(1) : B->dims(0);
    int64_t B_cols = transB ? B->dims(0) : B->dims(1);
    
    std::cout << "A.shape = (" << A->dims(0) << ", " << A->dims(1) << ")" << (transA ? " [transposed]" : "") << std::endl;
    std::cout << "B.shape = (" << B->dims(0) << ", " << B->dims(1) << ")" << (transB ? " [transposed]" : "") << std::endl;
    std::cout << "C.shape = (" << C->dims(0) << ")" << std::endl;
    std::cout << "Effective A: (" << A_rows << ", " << A_cols << "), B: (" << B_rows << ", " << B_cols << ")" << std::endl;
    
    // For matrix multiplication A * B: A_cols must equal B_rows
    if (A_cols != B_rows) {
        std::cerr << "Matrix dimension mismatch: A cols=" << A_cols << ", B rows=" << B_rows << std::endl;
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    
    // Bias vector C should match the number of output columns
    if (B_cols != C->dims(0)) {
        std::cerr << "Bias dimension mismatch: B cols=" << B_cols << ", C size=" << C->dims(0) << std::endl;
        throw std::invalid_argument("Bias dimensions are not compatible with the result in Gemm.");
    }

    // Calculate output dimensions for A * B
    int64_t output_rows = A_rows;
    int64_t output_cols = B_cols;
    int64_t inner_dim = A_cols;    std::cout << "Output dimensions: " << output_rows << " x " << output_cols << std::endl;

    // Create output tensor
    onnx::TensorProto* result = new onnx::TensorProto;
    result->set_data_type(onnx::TensorProto::FLOAT);
    result->add_dims(output_rows);
    result->add_dims(output_cols);

    // Allocate memory for output
    std::vector<float> outData(output_rows * output_cols);
    std::cout << "outData size: " << outData.size() << std::endl;

    // Get raw data pointers
    const float* AData = reinterpret_cast<const float*>(A->raw_data().data());
    const float* BData = reinterpret_cast<const float*>(B->raw_data().data());
    const float* CData = reinterpret_cast<const float*>(C->raw_data().data());

    std::cout << "Running gemm" << std::endl;
    
    // Expand bias vector to match output dimensions for GEMM operation
    std::vector<float> bias_expanded(output_rows * output_cols);
    for (int r = 0; r < output_rows; ++r) {
        for (int c = 0; c < output_cols; ++c) {
            bias_expanded[r * output_cols + c] = CData[c];
        }
    }
    
    // Call the GEMM function with transpose handling
    if (transB) {
        // If B is transposed, we need to handle it differently
        // For now, let's create a transposed copy of B
        std::vector<float> B_transposed(B->dims(0) * B->dims(1));
        for (int i = 0; i < B->dims(0); ++i) {
            for (int j = 0; j < B->dims(1); ++j) {
                B_transposed[j * B->dims(0) + i] = BData[i * B->dims(1) + j];
            }
        }
        gemm(AData, B_transposed.data(), bias_expanded.data(), outData.data(), output_rows, inner_dim, output_cols);
    } else {
        // No transpose needed
        gemm(AData, BData, bias_expanded.data(), outData.data(), output_rows, inner_dim, output_cols);
    }
    
    std::cout << "Finished gemm" << std::endl;

    // Set the raw data
    result->set_raw_data(outData.data(), sizeof(float) * outData.size());

    // Set output name
    if (node.output_size() != 1) {
        throw std::runtime_error("Gemm operator expects exactly one output tensor.");
    }
    result->set_name(node.output()[0]);

    // Print output values (first few for debugging)
    std::cout << "GEMM output: ";
    int print_limit = std::min(10, (int)outData.size());
    for (int i = 0; i < print_limit; ++i) {
        std::cout << outData[i] << ", ";
    }
    if (outData.size() > print_limit) std::cout << "...";
    std::cout << std::endl;

    return result;
}

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
onnx::TensorProto *flatten(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node)
{
    std::cout << "Op: Flatten" << std::endl;
    if (inputs.size() != 1)
    {
        exit(1);
    }

    const auto *inputTensor = inputs[0];

    int64_t axis = getFlattenAxis(node);
    int64_t dimBefore = std::accumulate(inputTensor->dims().begin(), inputTensor->dims().begin() + axis, 1, std::multiplies<int64_t>());
    int64_t dimAfter = std::accumulate(inputTensor->dims().begin() + axis, inputTensor->dims().end(), 1, std::multiplies<int64_t>());

    onnx::TensorProto *flattened = new onnx::TensorProto(*inputTensor);

    flattened->clear_dims();
    flattened->add_dims(dimBefore);
    flattened->add_dims(dimAfter);

    // Diagnostic printing
    int float_size = flattened->raw_data().size() / sizeof(float);
    std::cout << "flatten float_size: " << float_size << std::endl;
    std::cout << "flatten out:";
    const float *raw = (float *)flattened->raw_data().data();
    for (int i = 0; i < float_size; ++i)
    {
        std::cout << " " << raw[i];
    }
    std::cout << std::endl;

    // Set tensor name.
    assert(node.output_size() == 1);
    std::string out_name = node.output()[0];
    flattened->set_name(out_name.c_str()); // Set the input name (important for ONNX runtimes)
    std::cout << "Flattened name: " << flattened->name() << std::endl;
    return flattened;
}

// relu
onnx::TensorProto *relu(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node)
{
    std::cout << "Op: Relu" << std::endl;
    assert(inputs.size() == 1);
    const auto &inputTensor = *inputs[0];

    onnx::TensorProto *outputTensor = new onnx::TensorProto(inputTensor);

    if (outputTensor->data_type() == onnx::TensorProto::FLOAT && !outputTensor->raw_data().empty())
    {
        const float *inputData = reinterpret_cast<const float *>(outputTensor->raw_data().data());
        int numElements = outputTensor->raw_data().size() / sizeof(float);

        // Modify the outputTensor's raw_data in place
        float *outputData = reinterpret_cast<float *>(outputTensor->mutable_raw_data()->data());
        for (int i = 0; i < numElements; ++i)
        {
            outputData[i] = std::max(0.0f, inputData[i]);
        }

        // Print the modified output values (optional)
        std::cout << "ReLU: ";
        assert(outputTensor->dims_size() == 2);
        for (int i = 0; i < outputTensor->dims(1); ++i)
        { // Assuming the second dimension is the relevant one
            std::cout << outputData[i] << ", ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cerr << "Unsupported data type or empty raw data in ReLU" << std::endl;
        exit(1);
    }

    assert(node.output_size() == 1);
    outputTensor->set_name(node.output(0));
    std::cout << "ReLU name: " << outputTensor->name() << std::endl;

    return outputTensor;
}

float extract_const(onnx::NodeProto node)
{
    std::cout << "Const: " << node.output(0) << std::endl;
    const onnx::AttributeProto *value_attr = nullptr;
    for (const auto &attr : node.attribute())
    {
        if (attr.name() == "value")
        {
            value_attr = &attr;
            break;
        }
    }
    if (!value_attr)
    {
        std::cerr << "Expected constant to have value attr, got nothing" << std::endl;
        exit(1);
    }
    const onnx::TensorProto &tensor = value_attr->t();
    float floatValue;
    if (tensor.data_type() != onnx::TensorProto::FLOAT)
    {
        std::cerr << "Const is not of float type" << std::endl;
        exit(1);
    }

    if (tensor.dims_size() == 0)
    {
        // Const is a scalar
        floatValue = tensor.float_data(0); // Extract the float
        std::cout << "Constant value: " << floatValue << std::endl;
        return floatValue;
    }
    std::cerr << "Unimplemented handling for multi-dim consts" << std::endl;
    exit(1);
}

void printRawTensorData(const onnx::TensorProto &tensor)
{
    // Determine data type
    switch (tensor.data_type())
    {
    case onnx::TensorProto_DataType_FLOAT:
    {
        const float *data = tensor.float_data().data();
        for (int i = 0; i < tensor.float_data_size(); ++i)
        {
            std::cout << data[i] << " ";
        }
        break;
    }
    case onnx::TensorProto_DataType_INT32:
    {
        const int32_t *data = tensor.int32_data().data();
        for (int i = 0; i < tensor.int32_data_size(); ++i)
        {
            std::cout << data[i] << " ";
        }
        break;
    }
    // ... (Add cases for other data types: INT64, UINT8, etc.)
    default:
        std::cerr << "Unsupported data type: " << tensor.data_type() << std::endl;
    }
    std::cout << std::endl; // Newline after printing
}

int getFlattenAxis(const onnx::NodeProto &node)
{
    int axis = 1; // Default value (as per ONNX specification)

    for (const auto &attr : node.attribute())
    {
        if (attr.name() == "axis")
        {
            if (attr.has_i())
            {
                axis = attr.i(); // Get the 'axis' value
                break;
            }
            else
            {
                std::cerr << "Error: Flatten node has 'axis' attribute, but it's not an integer." << std::endl;
            }
        }
    }

    return axis;
}

// =================================================================
// SECTION: MNIST Classification Functions
// =================================================================

/**
 * @brief Applies softmax activation to convert logits to probabilities
 * 
 * Softmax function: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
 * This converts raw model outputs (logits) to a probability distribution.
 * 
 * @param logits Raw output values from the neural network
 * @return Vector of probabilities that sum to 1.0
 */
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    
    // Find maximum value for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp(x_i - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    
    // Normalize to get probabilities
    for (size_t i = 0; i < probabilities.size(); ++i) {
        probabilities[i] /= sum;
    }
    
    return probabilities;
}

/**
 * @brief Finds the index of the maximum value (predicted class)
 * 
 * @param probabilities Vector of probability values
 * @return Index of the maximum probability (0-9 for MNIST digits)
 */
int argmax(const std::vector<float>& probabilities) {
    return std::distance(probabilities.begin(), 
                        std::max_element(probabilities.begin(), probabilities.end()));
}

/**
 * @brief Classifies MNIST output and displays results
 * 
 * Takes the raw neural network output, applies softmax to get probabilities,
 * finds the predicted digit, and displays detailed classification results.
 * 
 * @param output_data Raw output data from the model
 * @param output_size Number of output classes (should be 10 for MNIST)
 */
void classify_mnist_output(const float* output_data, int output_size) {
    // Convert raw data to vector for easier processing
    std::vector<float> logits(output_data, output_data + output_size);
    
    // Apply softmax to get probabilities
    std::vector<float> probabilities = softmax(logits);
    
    // Find predicted class
    int predicted_digit = argmax(probabilities);
    float confidence = probabilities[predicted_digit];
    
    // Display results
    std::cout << "\n--- CLASSIFICATION RESULTS ---" << std::endl;
    std::cout << "Predicted digit: " << predicted_digit << std::endl;
    std::cout << "Confidence: " << std::fixed << std::setprecision(2) << (confidence * 100.0f) << "%" << std::endl;
    
    std::cout << "\nProbability distribution:" << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << "  Digit " << i << ": " << std::fixed << std::setprecision(4) 
                  << probabilities[i] << " (" << std::setprecision(1) 
                  << (probabilities[i] * 100.0f) << "%)" << std::endl;
    }
    
    // Visual confidence bar for the predicted digit
    std::cout << "\nConfidence bar for digit " << predicted_digit << ": ";
    int bar_length = (int)(confidence * 20); // Scale to 20 characters
    for (int i = 0; i < 20; ++i) {
        if (i < bar_length) {
            std::cout << "█";
        } else {
            std::cout << "░";
        }
    }
    std::cout << " " << std::setprecision(1) << (confidence * 100.0f) << "%" << std::endl;
}