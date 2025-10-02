# Simple benchmark setup - can be expanded later
add_executable(tnie_benchmark
    benchmark_main.cpp
)

target_link_libraries(tnie_benchmark 
    PRIVATE 
        tnie
)

target_include_directories(tnie_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
