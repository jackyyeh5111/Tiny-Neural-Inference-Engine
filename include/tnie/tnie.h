#pragma once

// Main header that includes all TNIE components
#include "tensor.h"
#include "operator.h"
#include "graph.h"

namespace tnie {

/**
 * @brief Get version information
 */
const char* get_version();

/**
 * @brief Initialize the library (if needed)
 */
void initialize();

/**
 * @brief Cleanup resources (if needed)
 */
void finalize();

} // namespace tnie
