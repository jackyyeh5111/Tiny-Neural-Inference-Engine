# Generate protobuf source files from .proto definition
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS onnx-ml.proto)

# Print paths to generated files for debugging
message("PROTO_SRCS = ${PROTO_SRCS}")
message("PROTO_HDRS = ${PROTO_HDRS}")

# Create the main inference engine executable
add_executable(inference_engine
    main.cpp
    gemm.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS})

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

# Add testing support if GTest is available
find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    
    # Create GEMM unit test executable
    add_executable(gemm_test test/gemm_test.cpp gemm.cpp)
    target_link_libraries(gemm_test GTest::GTest GTest::Main)
    add_test(NAME gemm_test COMMAND gemm_test)
    
    message("Tests enabled: GTest found")
else()
    message("Tests disabled: GTest not found")
endif()