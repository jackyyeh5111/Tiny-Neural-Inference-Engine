find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Download and build GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9fd7188ec8e5ef3f55e40e0.zip
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Test sources
set(TEST_SOURCES
    test_tensor.cpp
    test_operator.cpp
    test_graph.cpp
)

# Create test executable
add_executable(tnie_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(tnie_tests 
    PRIVATE 
        tnie
        gtest_main
        gtest
        gmock
)

# Include directories
target_include_directories(tnie_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(tnie_tests)
