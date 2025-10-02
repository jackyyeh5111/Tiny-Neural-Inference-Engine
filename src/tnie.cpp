#include "tnie/tnie.h"

namespace tnie {

const char* get_version() {
    return "0.1.0";
}

void initialize() {
    // Initialize any global state if needed
    // For now, this is a no-op
}

void finalize() {
    // Cleanup any global state if needed
    // For now, this is a no-op
}

} // namespace tnie
