###############################################################################
# PROJECT CONFIGURATION & GLOBAL SETTINGS
###############################################################################

cmake_minimum_required(VERSION 3.15)
project(EntropyMinimizer LANGUAGES CXX CUDA)

# =============================================================================
# GLOBAL COMPILATION SETTINGS (Applied to all targets)
# =============================================================================

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture
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
# PLATFORM DETECTION
# =============================================================================

if(APPLE)
    set(PLATFORM "apple")
elseif(UNIX)
    set(PLATFORM "linux")
else()
    message(FATAL_ERROR "Unsupported platform.")
endif()

# =============================================================================
# EXTERNAL DEPENDENCIES
# =============================================================================

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# NVTX (NVIDIA Tools Extension)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../NVTX" "ImportNVTX")

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
    set(OPENBLAS_ROOT "$ENV{HOME}/.local")
    
    # Find OpenBLAS and LAPACKE with NO_DEFAULT_PATH to prioritize local install
    find_library(OPENBLAS_LIB openblas PATHS ${OPENBLAS_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(LAPACKE_LIB lapacke PATHS ${OPENBLAS_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
    
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
# TARGET: moe (Main Optimization Engine)
# =============================================================================

create_executable(moe "${SRC_DIR}/main/moe")

# Link libraries for moe
target_link_libraries(moe PRIVATE
    # Minimizer libraries
    minimizer_algorithm
    minimizer_core
    minimizer_config
    minimizer_analysis
    minimizer_utilities
    
    # Channel libraries
    kraus_generators
    channel_utils
    
    # Utility libraries
    messaging
    serializer
    uuid
    
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

# =============================================================================
# TARGET: phase1_parallel (Parallel Phase 1 with Task Queue)
# =============================================================================
add_executable(phase1_parallel
    "${SRC_DIR}/main/phase1/phase1_parallel.cu"
)

target_link_libraries(phase1_parallel PRIVATE
    # Channel libraries
    kraus_generators
    channel_utils
    
    # Utility libraries
    messaging
    gpu_utilities
    
    # Minimizer libraries
    minimizer_analysis
    
    # Configuration
    yaml-cpp::yaml-cpp
    
    # CUDA libraries
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    
    # Profiling
    nvtx3-cpp
    
    # LAPACK backend
    ${LAPACK_LIBRARIES}
)

set_target_properties(phase1_parallel PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)


# =============================================================================
# TARGET: phase1 (Main Phase 1 Application)
# =============================================================================
add_executable(phase1
    "${SRC_DIR}/main/phase1/phase1.cu"
)

target_link_libraries(phase1 PRIVATE
    # Channel libraries
    kraus_generators
    channel_utils
    
    # Utility libraries
    messaging
    
    # Minimizer libraries
    minimizer_analysis
    
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
# TARGET: stage1 (Placeholder for your next executable)
# =============================================================================

# Uncomment and modify when ready to add stage1 executable
# create_executable(stage1 "${SRC_DIR}/main/stage1")
# 
# target_link_libraries(stage1 PRIVATE
#     # Add your libraries here
#     minimizer_algorithm
#     minimizer_core
#     # ... etc
#     ${LAPACK_LIBRARIES}
# )

###############################################################################
# BUILD SUMMARY
###############################################################################

message(STATUS "")
message(STATUS "========== BUILD CONFIGURATION SUMMARY ==========")
message(STATUS "Project:              ${PROJECT_NAME}")
message(STATUS "C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Architecture:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Platform:             ${PLATFORM}")
message(STATUS "LAPACK Backend:       ${LAPACK}")
message(STATUS "Build Type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "================================================")
message(STATUS "")

