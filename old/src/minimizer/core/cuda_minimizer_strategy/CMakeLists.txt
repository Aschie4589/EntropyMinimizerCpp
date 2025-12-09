# src/minimizer/core/cuda_minimizer_strategy/CMakeLists.txt
# Strategy pattern implementations for memory-efficient minimization

add_library(minimizer_strategies STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/high_memory_strategy.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/low_memory_strategy.cu
)

set_target_properties(minimizer_strategies PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_compile_features(minimizer_strategies PUBLIC cxx_std_17)

target_include_directories(minimizer_strategies
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/external
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(minimizer_strategies PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

message(STATUS "Minimizer strategies module configured: minimizer_strategies library")
