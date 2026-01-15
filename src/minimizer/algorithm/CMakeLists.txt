# Minimizer Algorithm Library
# Core algorithm execution and minimization strategies

add_library(minimizer_algorithm
    cuda_minimization_strategy.cu
    generic_minimization_strategy.cpp
    algorithm_manager.cpp
    strategy_factory.cpp
)

set_target_properties(minimizer_algorithm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(minimizer_algorithm PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(minimizer_algorithm PUBLIC
    minimizer_state
    compute_cuda
    compute_cpu
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

target_compile_features(minimizer_algorithm PUBLIC cxx_std_17)
