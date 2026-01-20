cmake_minimum_required(VERSION 3.18)

# GPU utilities library (shared to avoid linking issues)
add_library(gpu_utilities SHARED
    gpu_resource_manager.cu
    nvml_wrapper.cpp
)

target_include_directories(gpu_utilities PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find NVML library
find_library(NVML_LIBRARY NAMES nvidia-ml libnvidia-ml.so.1
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
    DOC "NVIDIA Management Library"
)

if(NOT NVML_LIBRARY)
    message(WARNING "NVML library not found. GPU utilities may not work properly.")
    set(NVML_LIBRARY "")
endif()

target_link_libraries(gpu_utilities PUBLIC
    ${CUDA_LIBRARIES}
    ${NVML_LIBRARY}
)

set_target_properties(gpu_utilities PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
