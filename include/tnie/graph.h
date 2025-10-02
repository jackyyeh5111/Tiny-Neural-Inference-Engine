#pragma once

#include "operator.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace tnie {

/**
 * @brief Graph node representing an operator and its connections
 */
struct GraphNode {
    std::unique_ptr<Operator> op;
    std::vector<size_t> input_ids;   // Indices of input tensors
    std::vector<size_t> output_ids;  // Indices of output tensors
    std::vector<size_t> dependencies; // Indices of dependent nodes
    
    GraphNode(std::unique_ptr<Operator> operator) 
        : op(std::move(operator)) {}
};

/**
 * @brief Simple graph executor for running operators with dependencies
 */
class GraphExecutor {
public:
    GraphExecutor() = default;
    ~GraphExecutor() = default;

    // Add operator to the graph
    size_t add_operator(std::unique_ptr<Operator> op,
                       const std::vector<size_t>& input_ids,
                       const std::vector<size_t>& output_ids);
    
    // Execute the graph
    void execute(std::vector<Tensor>& tensors);
    
    // Get execution order (topological sort)
    std::vector<size_t> get_execution_order() const;
    
    // Clear the graph
    void clear();
    
    // Graph information
    size_t num_nodes() const { return nodes_.size(); }
    const GraphNode& get_node(size_t idx) const { return nodes_[idx]; }

private:
    std::vector<GraphNode> nodes_;
    
    // Topological sort helper
    void dfs_visit(size_t node_idx, 
                  std::vector<bool>& visited,
                  std::vector<size_t>& order) const;
};

} // namespace tnie
