# src/minimizer/core/kernels/CMakeLists.txt
# CUDA kernel implementations for vector and matrix operations

add_library(minimizer_kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/rescale_vecs.cu
)

set_target_properties(minimizer_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_compile_features(minimizer_kernels PUBLIC cxx_std_17)

target_include_directories(minimizer_kernels
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(minimizer_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
)

message(STATUS "Minimizer kernels module configured: minimizer_kernels library")
