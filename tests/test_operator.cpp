#include <gtest/gtest.h>
#include "tnie/operator.h"

using namespace tnie;

class OperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test fixtures
    }
};

TEST_F(OperatorTest, GemmShapeInference) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    
    std::vector<Shape> input_shapes = {Shape({2, 3}), Shape({3, 4})};
    std::vector<Shape> output_shapes = gemm_op->infer_output_shapes(input_shapes);
    
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], Shape({2, 4}));
}

TEST_F(OperatorTest, GemmShapeInferenceTransposed) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm", 1.0f, 0.0f, true, false);
    
    std::vector<Shape> input_shapes = {Shape({3, 2}), Shape({3, 4})};
    std::vector<Shape> output_shapes = gemm_op->infer_output_shapes(input_shapes);
    
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], Shape({2, 4}));
}

TEST_F(OperatorTest, GemmBasicExecution) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    
    // A: 2x2 identity matrix
    Tensor A(Shape({2, 2}), {1.0f, 0.0f, 0.0f, 1.0f});
    // B: 2x2 matrix
    Tensor B(Shape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
    
    std::vector<Tensor> inputs = {A, B};
    std::vector<Tensor> outputs;
    
    gemm_op->forward(inputs, outputs);
    
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].shape(), Shape({2, 2}));
    
    // Result should be B since A is identity
    EXPECT_FLOAT_EQ(outputs[0][0], 2.0f);
    EXPECT_FLOAT_EQ(outputs[0][1], 3.0f);
    EXPECT_FLOAT_EQ(outputs[0][2], 4.0f);
    EXPECT_FLOAT_EQ(outputs[0][3], 5.0f);
}

TEST_F(OperatorTest, GemmWithScaling) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm", 2.0f, 0.0f);
    
    Tensor A(Shape({2, 2}), {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor B(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
    
    std::vector<Tensor> inputs = {A, B};
    std::vector<Tensor> outputs;
    
    gemm_op->forward(inputs, outputs);
    
    ASSERT_EQ(outputs.size(), 1);
    
    // Result should be 2 * B
    EXPECT_FLOAT_EQ(outputs[0][0], 2.0f);
    EXPECT_FLOAT_EQ(outputs[0][1], 4.0f);
    EXPECT_FLOAT_EQ(outputs[0][2], 6.0f);
    EXPECT_FLOAT_EQ(outputs[0][3], 8.0f);
}

TEST_F(OperatorTest, GemmInvalidInputs) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    
    // Mismatched dimensions
    Tensor A(Shape({2, 3}), std::vector<float>(6, 1.0f));
    Tensor B(Shape({2, 2}), std::vector<float>(4, 1.0f)); // Should be 3x2
    
    std::vector<Tensor> inputs = {A, B};
    std::vector<Tensor> outputs;
    
    EXPECT_THROW(gemm_op->forward(inputs, outputs), std::invalid_argument);
}

TEST_F(OperatorTest, Conv2DShapeInference) {
    auto conv_op = OperatorFactory::create_conv2d("test_conv", {3, 3});
    
    std::vector<Shape> input_shapes = {
        Shape({1, 3, 32, 32}), // batch=1, channels=3, height=32, width=32
        Shape({64, 3, 3, 3})   // out_channels=64, in_channels=3, kernel_h=3, kernel_w=3
    };
    
    std::vector<Shape> output_shapes = conv_op->infer_output_shapes(input_shapes);
    
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], Shape({1, 64, 30, 30})); // (32-3)/1+1 = 30
}

TEST_F(OperatorTest, Conv2DBasicExecution) {
    auto conv_op = OperatorFactory::create_conv2d("test_conv", {1, 1});
    
    // Simple 1x1 convolution (pointwise)
    Tensor input(Shape({1, 1, 2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor kernel(Shape({1, 1, 1, 1}), {2.0f}); // Scale by 2
    
    std::vector<Tensor> inputs = {input, kernel};
    std::vector<Tensor> outputs;
    
    conv_op->forward(inputs, outputs);
    
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].shape(), Shape({1, 1, 2, 2}));
    
    // Results should be input scaled by 2
    EXPECT_FLOAT_EQ(outputs[0][0], 2.0f);
    EXPECT_FLOAT_EQ(outputs[0][1], 4.0f);
    EXPECT_FLOAT_EQ(outputs[0][2], 6.0f);
    EXPECT_FLOAT_EQ(outputs[0][3], 8.0f);
}

TEST_F(OperatorTest, OperatorFactory) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    EXPECT_EQ(gemm_op->name(), "test_gemm");
    EXPECT_EQ(gemm_op->type(), "Gemm");
    
    auto conv_op = OperatorFactory::create_conv2d("test_conv", {3, 3});
    EXPECT_EQ(conv_op->name(), "test_conv");
    EXPECT_EQ(conv_op->type(), "Conv2D");
}
