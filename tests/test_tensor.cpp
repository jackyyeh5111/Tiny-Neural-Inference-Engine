#include <gtest/gtest.h>
#include "tnie/tensor.h"

using namespace tnie;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test fixtures
    }
};

TEST_F(TensorTest, ShapeCreation) {
    Shape shape({2, 3, 4});
    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape.size(), 24);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

TEST_F(TensorTest, TensorCreation) {
    Shape shape({2, 3});
    Tensor tensor(shape, DataType::FLOAT32);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(TensorTest, TensorWithData) {
    Shape shape({2, 2});
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor(shape, data);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor[0], 1.0f);
    EXPECT_EQ(tensor[1], 2.0f);
    EXPECT_EQ(tensor[2], 3.0f);
    EXPECT_EQ(tensor[3], 4.0f);
}

TEST_F(TensorTest, TensorCopy) {
    Shape shape({2, 2});
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor1(shape, data);
    
    Tensor tensor2 = tensor1; // Copy constructor
    
    EXPECT_EQ(tensor2.shape(), shape);
    EXPECT_EQ(tensor2[0], 1.0f);
    EXPECT_EQ(tensor2[1], 2.0f);
    
    // Modify original, copy should be unchanged
    tensor1[0] = 10.0f;
    EXPECT_EQ(tensor1[0], 10.0f);
    EXPECT_EQ(tensor2[0], 1.0f);
}

TEST_F(TensorTest, TensorMove) {
    Shape shape({2, 2});
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor1(shape, data);
    
    Tensor tensor2 = std::move(tensor1); // Move constructor
    
    EXPECT_EQ(tensor2.shape(), shape);
    EXPECT_EQ(tensor2[0], 1.0f);
    EXPECT_EQ(tensor2[1], 2.0f);
}

TEST_F(TensorTest, ZerosTensor) {
    Shape shape({3, 2});
    Tensor tensor = Tensor::zeros(shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 6);
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], 0.0f);
    }
}

TEST_F(TensorTest, OnesTensor) {
    Shape shape({2, 3});
    Tensor tensor = Tensor::ones(shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 6);
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], 1.0f);
    }
}

TEST_F(TensorTest, RandomTensor) {
    Shape shape({3, 3});
    Tensor tensor = Tensor::random(shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 9);
    
    // Check that values are in [0, 1] range
    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_GE(tensor[i], 0.0f);
        EXPECT_LE(tensor[i], 1.0f);
    }
}

TEST_F(TensorTest, FillOperation) {
    Shape shape({2, 2});
    Tensor tensor = Tensor::zeros(shape);
    
    tensor.fill(5.5f);
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], 5.5f);
    }
}

TEST_F(TensorTest, IndexAccess) {
    Shape shape({2, 3});
    Tensor tensor = Tensor::zeros(shape);
    
    tensor[0] = 1.0f;
    tensor[1] = 2.0f;
    tensor[5] = 6.0f;
    
    EXPECT_EQ(tensor[0], 1.0f);
    EXPECT_EQ(tensor[1], 2.0f);
    EXPECT_EQ(tensor[5], 6.0f);
}

TEST_F(TensorTest, OutOfBoundsAccess) {
    Shape shape({2, 2});
    Tensor tensor = Tensor::zeros(shape);
    
    EXPECT_THROW(tensor[4], std::out_of_range);
    EXPECT_THROW(tensor[100], std::out_of_range);
}

TEST_F(TensorTest, InvalidDataSize) {
    Shape shape({2, 2});
    std::vector<float> data = {1.0f, 2.0f}; // Wrong size
    
    EXPECT_THROW(Tensor(shape, data), std::invalid_argument);
}
