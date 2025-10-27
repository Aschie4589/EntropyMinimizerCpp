# src/channel/CMakeLists.txt
# Channel generation and utility functions for Kraus operator computation

# ==============================================================================
# UTILITIES LIBRARY
# ==============================================================================
# GPU utilities for generating Haar random unitaries via QR decomposition
add_library(channel_utils STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities/generate_haar_unitary.cu
)

set_target_properties(channel_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(channel_utils PUBLIC cxx_std_17)

target_include_directories(channel_utils
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(channel_utils PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cusolver
)

# ==============================================================================
# KRAUS GENERATORS LIBRARY
# ==============================================================================
# Kraus operator generation via Clebsch-Gordan coefficients or random unitaries
add_library(kraus_generators STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/generator/cg_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generator/random_generator.cu
)

# Mark random_generator.cu as CUDA source
set_source_files_properties(
    generator/random_generator.cu
    PROPERTIES
    LANGUAGE CUDA
)

set_target_properties(kraus_generators PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(kraus_generators PUBLIC cxx_std_17)

target_include_directories(kraus_generators
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(kraus_generators PUBLIC
    channel_utils
    clebsch
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cusolver
)

message(STATUS "Channel module configured: channel_utils and kraus_generators libraries")
