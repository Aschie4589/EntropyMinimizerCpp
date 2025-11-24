// Test runner for compute backend tests
#include <gtest/gtest.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Print CUDA device info for debugging
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess && deviceCount > 0) {
        std::cout << "=== CUDA Devices Found: " << deviceCount << " ===" << std::endl;
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "  Device " << i << ": " << prop.name 
                      << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
        }
    } else {
        std::cout << "=== No CUDA Devices Found - CPU-only tests ===" << std::endl;
    }
    
    // Run all tests
    return RUN_ALL_TESTS();
}
