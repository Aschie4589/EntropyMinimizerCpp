# Tests for compute backends

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/main/tests
    ${CUDAToolkit_INCLUDE_DIRS}
    ${LAPACK_INCLUDE_DIR}
)

# Collect all test source files
set(TEST_SOURCES
    main_test.cpp
    compute/memory_test.cpp
    compute/scratch_memory_test.cpp
    compute/svd_types_test.cpp
    compute/cuda_svd_solver_test.cpp
    compute/cpu_svd_solver_test.cpp
    compute/stream_test.cu
    compute/random_test.cpp
    compute/linalg_test.cpp
)

# Create test executable
add_executable(compute_tests ${TEST_SOURCES})

# Compile compute backend sources directly with the test executable
target_sources(compute_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaStream.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaRandom.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaLinearAlgebra.cu
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaSolver.cu
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaSVDSolver.cu
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cuda/CudaDevice.cu
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuStream.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuRandom.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuLinearAlgebra.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuSolver.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuDevice.cpp
    ${CMAKE_SOURCE_DIR}/src/compute/backends/cpu/CpuSVDSolver.cpp
)

# Link libraries
target_link_libraries(compute_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

# Set CUDA properties
set_target_properties(compute_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enable C++17
target_compile_features(compute_tests PRIVATE cxx_std_17)

# Register tests with CTest
gtest_discover_tests(compute_tests)

# Add a custom target to run tests with colored output
add_custom_target(run_tests
    COMMAND compute_tests --gtest_color=yes
    DEPENDS compute_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running compute backend tests..."
)
