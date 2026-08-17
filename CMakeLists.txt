###############################################################################
# PROJECT CONFIGURATION & GLOBAL SETTINGS
###############################################################################

cmake_minimum_required(VERSION 3.15)
project(EntropyMinimizer LANGUAGES CXX)


# Enable CUDA language support if a compiler is detected
include(CheckLanguage)

if(NOT CMAKE_CUDA_COMPILER)
    find_program(CUDA_NVCC_EXECUTABLE nvcc
        HINTS
            ENV CUDA_PATH
            /opt/cuda
        PATH_SUFFIXES bin
    )
    if(CUDA_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH
            "CUDA compiler" FORCE)
    endif()
endif()

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

# =============================================================================
# GLOBAL COMPILATION SETTINGS (Applied to all targets)
# =============================================================================

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture (has no effect if CUDA is not enabled)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Common compiler flags (optimization, native, vectorization)
add_compile_options(-march=native -O3 -ftree-vectorize -fdiagnostics-color=always)

# Use RPATH to embed library paths in executables
# This allows executables to find libraries without setting LD_LIBRARY_PATH
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_NAME_DIR "@rpath")

# Suppress RPATH warning about library conflicts
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Add include directory to all targets
include_directories(${INCLUDE_DIR})

# =============================================================================
# EXTERNAL DEPENDENCIES
# =============================================================================
include(FetchContent)

# CUDA and NVTX (NVIDIA Tools Extension)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")

    # CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)

    # NVTX (NVIDIA Tools Extension)
    FetchContent_Declare(
        NVTX
        GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
        GIT_TAG        dev
    )
    FetchContent_MakeAvailable(NVTX)

else()
    message(WARNING "No CUDA compiler detected. CUDA-dependent targets will not be built.")
endif()

#Yaml-cpp (for configuration parsing)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
    GIT_TAG        master
)
FetchContent_MakeAvailable(yaml-cpp)

# Argparse
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse
    GIT_TAG        master
)
FetchContent_MakeAvailable(argparse)

# Nlohmann JSON (for JSON parsing)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG        v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

# OpenMP (for parallelization)
find_package(OpenMP REQUIRED)

# Google Test (for unit testing)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
enable_testing()
include(GoogleTest)

# =============================================================================
# LAPACK BACKEND CONFIGURATION
# =============================================================================

if(NOT DEFINED LAPACK)
    set(LAPACK "openblas")
endif()

if(LAPACK STREQUAL "accelerate")
    if(NOT APPLE)
        message(FATAL_ERROR "Accelerate framework is only available on macOS.")
    endif()
    message(STATUS "LAPACK Backend: Accelerate (macOS)")
    set(LAPACK_FOUND TRUE)
    set(LAPACK_LIBRARIES "-framework Accelerate")
    add_compile_definitions(LAPACK_ACCELERATE ACCELERATE_NEW_LAPACK)

elseif(LAPACK STREQUAL "openblas")   
    # Find OpenBLAS and LAPACKE with NO_DEFAULT_PATH to prioritize local install
    find_library(OPENBLAS_LIB openblas REQUIRED)
    find_library(LAPACKE_LIB lapacke REQUIRED)

    message(STATUS "LAPACK Backend: OpenBLAS")
    message(STATUS "  OpenBLAS: ${OPENBLAS_LIB}")
    message(STATUS "  LAPACKE:  ${LAPACKE_LIB}")
    
    set(LAPACK_FOUND TRUE)
    set(LAPACK_LIBRARIES ${OPENBLAS_LIB} ${LAPACKE_LIB} pthread m)
    set(LAPACK_INCLUDE_DIR ${OPENBLAS_ROOT}/include)
    add_compile_definitions(LAPACK_OPENBLAS)
    
    # Make LAPACK variables available to all targets
    set(LAPACK_OPENBLAS TRUE CACHE BOOL "Using OpenBLAS backend" FORCE)

else()
    message(FATAL_ERROR "Invalid LAPACK backend: ${LAPACK}. Choose from: accelerate, openblas")
endif()

# Cache LAPACK variables for use by subdirectories
set(LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" CACHE STRING "LAPACK libraries" FORCE)
set(LAPACK_INCLUDE_DIR "${LAPACK_INCLUDE_DIR}" CACHE STRING "LAPACK include directory" FORCE)

# Configure RPATH for executables to find libraries
if(LAPACK STREQUAL "openblas")
    set(CMAKE_BUILD_RPATH "${OPENBLAS_ROOT}/lib")
    message(STATUS "RPATH configured to: ${CMAKE_BUILD_RPATH}")
endif()

# =============================================================================
# BUILD ALL LIBRARIES (src tree)
# =============================================================================

add_subdirectory(src)

###############################################################################
# EXECUTABLE TARGETS
###############################################################################

# Define a helper function to simplify executable creation
function(create_executable TARGET_NAME MAIN_SOURCE_DIR)
    file(GLOB_RECURSE SOURCES "${MAIN_SOURCE_DIR}/*.cpp" "${MAIN_SOURCE_DIR}/*.cu")
    
    add_executable(${TARGET_NAME} ${SOURCES})
    
    # Directories
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CUDAToolkit_INCLUDE_DIRS} 
        ${INCLUDE_DIR} 
        ${SRC_DIR}/external
        ${LAPACK_INCLUDE_DIR}
    )
    
    # CUDA settings
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    message(STATUS "Executable target configured: ${TARGET_NAME}")
endfunction()

# =============================================================================
# TARGET: moe
# =============================================================================
# TEMPORARILY DISABLED - needs update to new API
# create_executable(moe "${SRC_DIR}/main/moe")

# Link libraries for moe
# target_link_libraries(moe PRIVATE
    # Minimizer libraries
#     minimizer_algorithm
#     minimizer_core
#     minimizer_config
#     minimizer_analysis
#     minimizer_utilities
    
    # Channel libraries
#     kraus_generators
#     channel_utils
    
    # Utility libraries
#     messaging
#     serializer
#     uuid
    
    # CUDA libraries
#     CUDA::cudart
#     CUDA::cublas
#     CUDA::cusolver
#     CUDA::curand
    
    # Profiling
#     nvtx3-cpp
    
    # Parallelization
#     OpenMP::OpenMP_CXX
    
    # LAPACK backend
#     ${LAPACK_LIBRARIES}
# )



# =============================================================================
# TARGET: phase1 (Main Phase 1 Application)
# =============================================================================
# TEMPORARILY DISABLED - needs update to new API (missing tensor_entropy.h)
add_executable(phase1
    "${SRC_DIR}/main/phase1/phase1.cu"
    "${SRC_DIR}/main/phase1/tensor_entropy.cu"
)

target_include_directories(phase1 PRIVATE
    "${SRC_DIR}/main/phase1"
    "${CMAKE_SOURCE_DIR}/src/external"
)

target_link_libraries(phase1 PRIVATE
    # Channel libraries
    kraus_generators
    channel_utils
    
    # Utility libraries
    messaging
    serializer
    gpu_utilities
    
    # Minimizer libraries
    minimizer_orchestration
    
    # Configuration
    yaml-cpp::yaml-cpp
    
    # CUDA libraries
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    
    # Profiling
    nvtx3-cpp
    
    # Parallelization
    OpenMP::OpenMP_CXX
    
    # LAPACK backend
    ${LAPACK_LIBRARIES}
)

set_target_properties(phase1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# =============================================================================
# TARGET: my_tests (Main Tests Application)
# =============================================================================

create_executable(my_tests "${SRC_DIR}/main/my_tests")

target_link_libraries(my_tests PRIVATE
    # Minimizer libraries (orchestration brings in most dependencies transitively)
    minimizer_orchestration
    
    # Compute libraries
    compute_cuda
    compute_cpu
    
    # Channel libraries
    kraus_generators
    channel_utils
    
    # Utility libraries
    messaging
    gpu_utilities
    
    # CUDA libraries
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    
    # Profiling
    nvtx3-cpp
    
    # Parallelization
    OpenMP::OpenMP_CXX
    
    # LAPACK backend
    ${LAPACK_LIBRARIES}
)




###############################################################################
# BUILD SUMMARY
###############################################################################

message(STATUS "")
message(STATUS "========== BUILD CONFIGURATION SUMMARY ==========")
message(STATUS "Project:              ${PROJECT_NAME}")
message(STATUS "C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Architecture:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "LAPACK Backend:       ${LAPACK}")
message(STATUS "Build Type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "================================================")
message(STATUS "")

