### This file contains project-wide configuration: compiler, cpp standard, dependencies, top level subdirectory includes.


cmake_minimum_required(VERSION 3.15)
project(moe LANGUAGES CXX CUDA)

# C++ standard and common flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Skip RPATH to allow local LAPACK/BLAS libraries to take precedence
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# Add base compiler flags
add_compile_options(-march=native -O3 -ftree-vectorize -fdiagnostics-color=always)

# Set CUDA architecture (optional but recommended)
# Replace 'sm_70' with your GPU architecture (e.g., sm_75, sm_80)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Dependencies
find_package(CUDAToolkit REQUIRED)

# Add source tree - this builds all libraries
add_subdirectory(src)

# Directories
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

include_directories(${INCLUDE_DIR})

# LAPACK implementation choice (default = openblas)
if(NOT DEFINED LAPACK)
    set(LAPACK "openblas")
endif()

# Platform check
if(APPLE)
    set(PLATFORM "apple")
elseif(UNIX)
    set(PLATFORM "linux")
else()
    message(FATAL_ERROR "Unsupported platform.")
endif()

# Find NVTX (NVIDIA Tools Extension)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../NVTX" "ImportNVTX")

# Collect only the main entry point source files
file(GLOB_RECURSE MAIN_SOURCES "${SRC_DIR}/main/moe/*.cpp" "${SRC_DIR}/main/moe/*.cu")

# Define the executable with only the main entry point
add_executable(moe ${MAIN_SOURCES})

target_include_directories(moe PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${INCLUDE_DIR} ${SRC_DIR}/external)

# Link all minimizer libraries
target_link_libraries(moe PRIVATE
    minimizer_algorithm
    minimizer_core
    minimizer_config
    minimizer_analysis
    minimizer_utilities
)

# Link channel libraries
target_link_libraries(moe PRIVATE
    kraus_generators
    channel_utils
)

# Link utility libraries
target_link_libraries(moe PRIVATE
    messaging
    serializer
    uuid
)

# Link CUDA libraries: cuSOLVER, cuRAND
target_link_libraries(moe PRIVATE
    CUDA::cudart
    CUDA::cusolver
    CUDA::curand
    CUDA::cublas
)

# Link against NVTX (nvToolsExt)
target_link_libraries(moe PRIVATE nvtx3-cpp)

# Set properties for CUDA compilation
set_target_properties(moe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(moe PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link and include LAPACK backend
if(LAPACK STREQUAL "accelerate")
    if(APPLE)
        add_compile_definitions(LAPACK_ACCELERATE ACCELERATE_NEW_LAPACK)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        # Link LAPACK
        target_link_libraries(moe PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(moe PRIVATE LAPACK_ACCELERATE)
    else()
        message(FATAL_ERROR "Accelerate is only available on macOS.")
    endif()

elseif(LAPACK STREQUAL "openblas")
    # Set your local OpenBLAS install path
    set(OPENBLAS_ROOT "$ENV{HOME}/.local")

    # Include headers and link directories - prioritize local over system
    target_include_directories(moe PRIVATE ${OPENBLAS_ROOT}/include)
    link_directories(${OPENBLAS_ROOT}/lib)

    # Manually find and link the static or shared libs
    # NO_DEFAULT_PATH ensures we use local libraries first, not system ones
    find_library(OPENBLAS_LIB openblas PATHS ${OPENBLAS_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(LAPACKE_LIB lapacke PATHS ${OPENBLAS_ROOT}/lib NO_DEFAULT_PATH REQUIRED)

    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    message(STATUS "Found LAPACKE: ${LAPACKE_LIB}")

    # Link libraries with full paths to avoid system library conflicts
    target_link_libraries(moe PRIVATE ${OPENBLAS_LIB} ${LAPACKE_LIB} pthread m)

    # Define macro for your code
    target_compile_definitions(moe PRIVATE LAPACK_OPENBLAS)

else()
    message(FATAL_ERROR "Invalid LAPACK setting: ${LAPACK}. Choose from: accelerate, openblas")
endif()

message(STATUS "MOE executable target configured successfully")

