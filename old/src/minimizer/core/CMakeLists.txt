# src/minimizer/core/CMakeLists.txt
# Core GPU-accelerated minimizer implementation

# Add subdirectories for strategies and kernels
add_subdirectory(cuda_minimizer_strategy)
add_subdirectory(kernels)

# Create the core minimizer library
add_library(minimizer_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_minimizer.cu
)

# Set CUDA separable compilation for proper linking
set_target_properties(minimizer_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_compile_features(minimizer_core PUBLIC cxx_std_17)

target_include_directories(minimizer_core
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/external
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(minimizer_core PUBLIC
    minimizer_strategies
    minimizer_kernels
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

message(STATUS "Minimizer core module configured: minimizer_core library")
