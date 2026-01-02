#include "config.h"

namespace tiny_engine {

    class InferEngine {
      public:
        InferEngine();

        bool load_model(const std::string& filename);

        void set_input(TensorPtr input_tensor);

        void run();

        const onnx::TensorProto* get_output(const std::string& name);

      private:
        void _process_results(const std::string& output_name);
        void _display_mnist_classification(const std::vector<float>& logits);
        void _register_operators();

        onnx::ModelProto model_;
        std::map<std::string, const onnx::TensorProto*> weights_map_;
        std::map<std::string, TensorPtr> intermediate_tensors_;  // Owns memory of computed tensors
        TensorPtr input_storage_;
        std::map<std::string, OperatorFunc> op_registry_;
    };

}  // namespace tiny_engine