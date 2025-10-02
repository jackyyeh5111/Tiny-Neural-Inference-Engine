#include <gtest/gtest.h>
#include "tnie/graph.h"

using namespace tnie;

class GraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        executor.clear();
    }
    
    GraphExecutor executor;
};

TEST_F(GraphTest, EmptyGraph) {
    EXPECT_EQ(executor.num_nodes(), 0);
    
    std::vector<Tensor> tensors;
    executor.execute(tensors); // Should not crash
}

TEST_F(GraphTest, SingleOperator) {
    auto gemm_op = OperatorFactory::create_gemm("gemm1");
    
    size_t node_id = executor.add_operator(std::move(gemm_op), {0, 1}, {2});
    
    EXPECT_EQ(executor.num_nodes(), 1);
    EXPECT_EQ(node_id, 0);
    
    const GraphNode& node = executor.get_node(0);
    EXPECT_EQ(node.input_ids.size(), 2);
    EXPECT_EQ(node.output_ids.size(), 1);
    EXPECT_EQ(node.dependencies.size(), 0); // No dependencies
}

TEST_F(GraphTest, TwoOperatorsWithDependency) {
    auto gemm1 = OperatorFactory::create_gemm("gemm1");
    auto gemm2 = OperatorFactory::create_gemm("gemm2");
    
    size_t node1 = executor.add_operator(std::move(gemm1), {0, 1}, {2});
    size_t node2 = executor.add_operator(std::move(gemm2), {2, 3}, {4});
    
    EXPECT_EQ(executor.num_nodes(), 2);
    
    // Check dependencies
    const GraphNode& node1_ref = executor.get_node(node1);
    const GraphNode& node2_ref = executor.get_node(node2);
    
    EXPECT_EQ(node1_ref.dependencies.size(), 0);
    EXPECT_EQ(node2_ref.dependencies.size(), 1);
    EXPECT_EQ(node2_ref.dependencies[0], node1);
}

TEST_F(GraphTest, ExecutionOrder) {
    auto gemm1 = OperatorFactory::create_gemm("gemm1");
    auto gemm2 = OperatorFactory::create_gemm("gemm2");
    auto gemm3 = OperatorFactory::create_gemm("gemm3");
    
    // Create a chain: gemm1 -> gemm2 -> gemm3
    size_t node1 = executor.add_operator(std::move(gemm1), {0, 1}, {2});
    size_t node2 = executor.add_operator(std::move(gemm2), {2, 3}, {4});
    size_t node3 = executor.add_operator(std::move(gemm3), {4, 5}, {6});
    
    std::vector<size_t> order = executor.get_execution_order();
    
    ASSERT_EQ(order.size(), 3);
    EXPECT_EQ(order[0], node1);
    EXPECT_EQ(order[1], node2);
    EXPECT_EQ(order[2], node3);
}

TEST_F(GraphTest, ParallelOperators) {
    auto gemm1 = OperatorFactory::create_gemm("gemm1");
    auto gemm2 = OperatorFactory::create_gemm("gemm2");
    
    // Two independent operations
    size_t node1 = executor.add_operator(std::move(gemm1), {0, 1}, {2});
    size_t node2 = executor.add_operator(std::move(gemm2), {3, 4}, {5});
    
    EXPECT_EQ(executor.num_nodes(), 2);
    
    // Both should have no dependencies
    const GraphNode& node1_ref = executor.get_node(node1);
    const GraphNode& node2_ref = executor.get_node(node2);
    
    EXPECT_EQ(node1_ref.dependencies.size(), 0);
    EXPECT_EQ(node2_ref.dependencies.size(), 0);
    
    std::vector<size_t> order = executor.get_execution_order();
    ASSERT_EQ(order.size(), 2);
    // Order can be either [0,1] or [1,0] since they're independent
}

TEST_F(GraphTest, SimpleExecution) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    
    executor.add_operator(std::move(gemm_op), {0, 1}, {2});
    
    // Prepare input tensors
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor(Shape({2, 2}), {1.0f, 0.0f, 0.0f, 1.0f})); // Identity matrix
    tensors.push_back(Tensor(Shape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f})); // Input matrix
    
    // Execute
    executor.execute(tensors);
    
    // Check output
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[2].shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(tensors[2][0], 2.0f);
    EXPECT_FLOAT_EQ(tensors[2][1], 3.0f);
    EXPECT_FLOAT_EQ(tensors[2][2], 4.0f);
    EXPECT_FLOAT_EQ(tensors[2][3], 5.0f);
}

TEST_F(GraphTest, ChainedExecution) {
    auto gemm1 = OperatorFactory::create_gemm("gemm1");
    auto gemm2 = OperatorFactory::create_gemm("gemm2");
    
    executor.add_operator(std::move(gemm1), {0, 1}, {2});
    executor.add_operator(std::move(gemm2), {2, 3}, {4});
    
    // Prepare input tensors
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::ones(Shape({2, 3}))); // tensor 0
    tensors.push_back(Tensor::ones(Shape({3, 2}))); // tensor 1
    tensors.push_back(Tensor());                    // tensor 2 (will be created)
    tensors.push_back(Tensor::ones(Shape({2, 1}))); // tensor 3
    
    // Execute
    executor.execute(tensors);
    
    // Check that execution completed and tensors were resized appropriately
    ASSERT_GE(tensors.size(), 5);
    EXPECT_EQ(tensors[2].shape(), Shape({2, 2})); // Result of first GEMM
    EXPECT_EQ(tensors[4].shape(), Shape({2, 1})); // Result of second GEMM
}

TEST_F(GraphTest, ClearGraph) {
    auto gemm_op = OperatorFactory::create_gemm("test_gemm");
    executor.add_operator(std::move(gemm_op), {0, 1}, {2});
    
    EXPECT_EQ(executor.num_nodes(), 1);
    
    executor.clear();
    
    EXPECT_EQ(executor.num_nodes(), 0);
}
