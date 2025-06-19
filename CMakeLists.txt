cmake_minimum_required(VERSION 3.15)
project(moe LANGUAGES CXX CUDA)

# C++ standard and common flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add base compiler flags
add_compile_options(-march=native -O3 -ftree-vectorize -fdiagnostics-color=always)

# Set CUDA architecture (optional but recommended)
# Replace 'sm_70' with your GPU architecture (e.g., sm_75, sm_80)
set(CMAKE_CUDA_ARCHITECTURES 89)


find_package(CUDAToolkit REQUIRED)
# Directories
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

include_directories(${INCLUDE_DIR})

# Collect all source files
file(GLOB_RECURSE CPP_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${SRC_DIR}/*.cu")

set(SOURCES ${CPP_SOURCES} ${CU_SOURCES})

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

# Define the executable
add_executable(moe ${SOURCES})

# Link CUDA libraries: cuSOLVER, cuRAND
target_link_libraries(moe PRIVATE
    CUDA::cudart
    CUDA::cusolver
    CUDA::curand
)

# Set properties for CUDA compilation (optional)
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

    # Include headers and link directories
    target_include_directories(moe PRIVATE ${OPENBLAS_ROOT}/include)
    link_directories(${OPENBLAS_ROOT}/lib)

    # Manually find and link the static or shared libs
    find_library(OPENBLAS_LIB openblas PATHS ${OPENBLAS_ROOT}/lib REQUIRED)
    find_library(LAPACKE_LIB lapacke PATHS ${OPENBLAS_ROOT}/lib REQUIRED)

    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    message(STATUS "Found LAPACKE: ${LAPACKE_LIB}")

    target_link_libraries(moe PRIVATE ${OPENBLAS_LIB} ${LAPACKE_LIB} pthread m)

    # Define macro for your code
    target_compile_definitions(moe PRIVATE LAPACK_OPENBLAS)

else()
    message(FATAL_ERROR "Invalid LAPACK setting: ${LAPACK}. Choose from: accelerate, openblas")
endif()



