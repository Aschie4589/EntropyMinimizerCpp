# Compute library CMakeLists.txt

# Find required packages
find_package(CUDAToolkit REQUIRED)

# CUDA backend sources
set(CUDA_BACKEND_SOURCES
    backends/cuda/CudaLinearAlgebra.cu
    backends/cuda/CudaSolver.cu
    backends/cuda/CudaSVDSolver.cu
    backends/cuda/CudaDevice.cu
    backends/cuda/CudaMemory.cpp
    backends/cuda/CudaRandom.cpp
    backends/cuda/CudaStream.cpp
    device/DeviceFactory.cu
)

# CPU backend sources  
set(CPU_BACKEND_SOURCES
    backends/cpu/CpuLinearAlgebra.cpp
    backends/cpu/CpuSolver.cpp
    backends/cpu/CpuSVDSolver.cpp
    backends/cpu/CpuDevice.cpp
    backends/cpu/CpuMemory.cpp
    backends/cpu/CpuRandom.cpp
    backends/cpu/CpuStream.cpp
)

# Accelerate backend sources (Apple only)
if(APPLE)
    set(ACCELERATE_BACKEND_SOURCES
        backends/accelerate/AccelerateLinearAlgebra.cpp
        backends/accelerate/AccelerateSolver.cpp
    )
    find_library(ACCELERATE_FRAMEWORK Accelerate)
endif()

# Create CUDA compute library
add_library(compute_cuda STATIC ${CUDA_BACKEND_SOURCES})
target_include_directories(compute_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(compute_cuda PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)
set_target_properties(compute_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Create CPU compute library
add_library(compute_cpu STATIC ${CPU_BACKEND_SOURCES})
target_include_directories(compute_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LAPACK_INCLUDE_DIR}
)

# Find BLAS/LAPACK for CPU backend
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(compute_cpu PUBLIC
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Create Accelerate compute library (Apple only)
if(APPLE)
    add_library(compute_accelerate STATIC ${ACCELERATE_BACKEND_SOURCES})
    target_include_directories(compute_accelerate PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(compute_accelerate PUBLIC
        ${ACCELERATE_FRAMEWORK}
    )
endif()

# Combined compute library for convenience
add_library(compute INTERFACE)
target_link_libraries(compute INTERFACE
    compute_cuda
    compute_cpu
)
if(APPLE)
    target_link_libraries(compute INTERFACE compute_accelerate)
endif()
