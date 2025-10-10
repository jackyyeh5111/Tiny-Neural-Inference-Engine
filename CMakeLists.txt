cmake_minimum_required(VERSION 3.10)
project(inference_engine)

# Set build type to Release for better performance (change to Debug for debugging)
set(CMAKE_BUILD_TYPE Release)

# Enable C++17 standard for modern C++ features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required dependencies
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src)
