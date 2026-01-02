#include "InferEnginer.h"
#include "onnx-ml.pb.h"
#include "utils.h"

using namespace tiny_engine;

int main(int argc, char* argv[]) {
    if (argc < 3)
        return 1;

    InferEngine engine;
    if (!engine.load_model(argv[1]))
        return -1;

    // Load Input (Simplified wrapper)
    TensorPtr input_tensor = utils::read_input(argv[2]);
    engine.set_input(std::move(input_tensor));

    engine.run();

    return 0;
}