# Generate protobuf source files from .proto definition
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS onnx-ml.proto)

# Print paths to generated files for debugging
message("PROTO_SRCS = ${PROTO_SRCS}")
message("PROTO_HDRS = ${PROTO_HDRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fno-omit-frame-pointer -fno-inline")
file(GLOB local_src *.cpp)

# Create the main inference engine executable
add_executable(inference_engine
    ${local_src}
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

# Include directories for protobuf and generated headers
target_include_directories(inference_engine
    PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Link against protobuf libraries
target_link_libraries(inference_engine
    PUBLIC
    ${PROTOBUF_LIBRARIES}
)
