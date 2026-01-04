#include "InferEnginer.h"
#include "onnx-ml.pb.h"
#include "utils.h"

#include <chrono>
#include <iomanip>

using namespace tiny_engine;

int main(int argc, char* argv[]) {
    if (argc < 3)
        return 1;

    InferEngine engine;
    if (!engine.load_model(argv[1]))
        return -1;

    // Load Input (Simplified wrapper)
    TensorPtr input_tensor = utils::read_mnist_input(argv[2]);
    engine.set_input(std::move(input_tensor));

    auto start = std::chrono::high_resolution_clock::now();

    // run
    engine.run();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "\n Total infer time: " << std::fixed << std::setprecision(3) << duration.count()
              << " ms" << std::endl;

    return 0;
}