# src/minimizer/utilities/CMakeLists.txt
# Helper utilities for the minimizer module

add_library(minimizer_utilities STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_random_vector.cu
)

set_target_properties(minimizer_utilities PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_compile_features(minimizer_utilities PUBLIC cxx_std_17)

target_include_directories(minimizer_utilities
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(minimizer_utilities PUBLIC
    CUDA::cudart
    CUDA::curand
)

message(STATUS "Minimizer utilities module configured: minimizer_utilities library")