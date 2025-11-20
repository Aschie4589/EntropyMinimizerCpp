cmake_minimum_required(VERSION 3.18)

# GPU utilities library (shared to avoid linking issues)
add_library(gpu_utilities SHARED
    gpu_resource_manager.cu
)

target_include_directories(gpu_utilities PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpu_utilities PUBLIC
    ${CUDA_LIBRARIES}
)

set_target_properties(gpu_utilities PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

