#include "tnie/graph.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace tnie {

size_t GraphExecutor::add_operator(std::unique_ptr<Operator> op,
                                  const std::vector<size_t>& input_ids,
                                  const std::vector<size_t>& output_ids) {
    size_t node_id = nodes_.size();
    nodes_.emplace_back(std::move(op));
    
    GraphNode& node = nodes_.back();
    node.input_ids = input_ids;
    node.output_ids = output_ids;
    
    // Calculate dependencies based on data flow
    for (size_t i = 0; i < node_id; ++i) {
        const GraphNode& other = nodes_[i];
        
        // Check if this node depends on the other node
        // (if any of this node's inputs are produced by the other node)
        for (size_t input_id : input_ids) {
            if (std::find(other.output_ids.begin(), other.output_ids.end(), input_id) 
                != other.output_ids.end()) {
                node.dependencies.push_back(i);
                break;
            }
        }
    }
    
    return node_id;
}

void GraphExecutor::execute(std::vector<Tensor>& tensors) {
    std::vector<size_t> execution_order = get_execution_order();
    
    for (size_t node_idx : execution_order) {
        const GraphNode& node = nodes_[node_idx];
        
        // Prepare input tensors
        std::vector<Tensor> inputs;
        for (size_t input_id : node.input_ids) {
            if (input_id >= tensors.size()) {
                throw std::runtime_error("Input tensor index out of bounds");
            }
            inputs.push_back(tensors[input_id]);
        }
        
        // Prepare output tensors
        std::vector<Tensor> outputs;
        for (size_t output_id : node.output_ids) {
            if (output_id < tensors.size()) {
                outputs.push_back(tensors[output_id]);
            }
        }
        
        // Execute the operator
        node.op->forward(inputs, outputs);
        
        // Update tensors vector with new outputs
        for (size_t i = 0; i < node.output_ids.size(); ++i) {
            size_t output_id = node.output_ids[i];
            if (output_id >= tensors.size()) {
                tensors.resize(output_id + 1);
            }
            if (i < outputs.size()) {
                tensors[output_id] = std::move(outputs[i]);
            }
        }
    }
}

std::vector<size_t> GraphExecutor::get_execution_order() const {
    std::vector<size_t> order;
    std::vector<bool> visited(nodes_.size(), false);
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (!visited[i]) {
            dfs_visit(i, visited, order);
        }
    }
    
    // Reverse to get correct topological order
    std::reverse(order.begin(), order.end());
    return order;
}

void GraphExecutor::clear() {
    nodes_.clear();
}

void GraphExecutor::dfs_visit(size_t node_idx, 
                             std::vector<bool>& visited,
                             std::vector<size_t>& order) const {
    visited[node_idx] = true;
    
    // Visit all dependencies first
    for (size_t dep_idx : nodes_[node_idx].dependencies) {
        if (!visited[dep_idx]) {
            dfs_visit(dep_idx, visited, order);
        }
    }
    
    order.push_back(node_idx);
}

} // namespace tnie
