# Source files
set(TNIE_SOURCES
    tensor.cpp
    operator.cpp
    graph.cpp
    tnie.cpp
)

# Create the main library
add_library(tnie ${TNIE_SOURCES})

# Link threading library
target_link_libraries(tnie PRIVATE Threads::Threads)

# Set target properties
set_target_properties(tnie PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(tnie
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create main executable
add_executable(tnie_main main.cpp)
target_link_libraries(tnie_main PRIVATE tnie)

# Installation
install(TARGETS tnie tnie_main
    EXPORT tnie-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
