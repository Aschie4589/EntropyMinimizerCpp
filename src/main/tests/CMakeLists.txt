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
    compute/multi_gpu_test.cu
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
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

# ============================================================================
# Minimizer Conditions Tests (header-only, no CUDA required)
# ============================================================================

add_executable(conditions_tests
    minimizer/conditions_test.cpp
)

target_link_libraries(conditions_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    minimizer_state
    minimizer_stopping
)

target_compile_features(conditions_tests PRIVATE cxx_std_17)

# Register conditions tests with CTest
gtest_discover_tests(conditions_tests)

# Add to run_tests target
add_custom_target(run_conditions_tests
    COMMAND conditions_tests --gtest_color=yes
    DEPENDS conditions_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running minimizer conditions tests..."
)

# ============================================================================
# Minimizer Core Tests (EntropyManager, etc.)
# ============================================================================

add_executable(core_tests
    minimizer/entropy_manager_test.cpp
)

target_link_libraries(core_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    minimizer_state
)

target_compile_features(core_tests PRIVATE cxx_std_17)

# Register core tests with CTest
gtest_discover_tests(core_tests)

# Add to run_tests target
add_custom_target(run_core_tests
    COMMAND core_tests --gtest_color=yes
    DEPENDS core_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running minimizer core tests..."
)

# ============================================================================
# Prediction Strategy Tests
# ============================================================================

add_executable(prediction_tests
    minimizer/exponential_fitting_strategy_test.cpp
    minimizer/linear_extrapolation_strategy_test.cpp
    minimizer/entropy_predictor_test.cpp
)

target_link_libraries(prediction_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    minimizer_state
    minimizer_prediction
)

target_compile_features(prediction_tests PRIVATE cxx_std_17)

# Register prediction tests with CTest
gtest_discover_tests(prediction_tests)

# Add to run_tests target
add_custom_target(run_prediction_tests
    COMMAND prediction_tests --gtest_color=yes
    DEPENDS prediction_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running entropy prediction tests..."
)

# ============================================================================
# Minimization Strategy Tests (interface tests, no device implementation)
# ============================================================================

add_executable(strategy_tests
    minimizer/core/test_minimization_strategy.cpp
)

target_link_libraries(strategy_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)

target_compile_features(strategy_tests PRIVATE cxx_std_17)

# Register strategy tests with CTest
gtest_discover_tests(strategy_tests)

# Add to run_tests target
add_custom_target(run_strategy_tests
    COMMAND strategy_tests --gtest_color=yes
    DEPENDS strategy_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running minimization strategy tests..."
)

# ============================================================================
# CUDA Minimization Strategy Tests (refactored for stateless API)
# ============================================================================

add_executable(cuda_strategy_tests
    minimizer/core/test_cuda_strategy.cpp
)

# Compile device sources directly
target_sources(cuda_strategy_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(cuda_strategy_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

set_target_properties(cuda_strategy_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(cuda_strategy_tests PRIVATE cxx_std_17)

# Register cuda strategy tests with CTest
gtest_discover_tests(cuda_strategy_tests)

# Add to run_tests target
add_custom_target(run_cuda_strategy_tests
    COMMAND cuda_strategy_tests --gtest_color=yes
    DEPENDS cuda_strategy_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CUDA minimization strategy tests..."
)

# ============================================================================
# Generic Minimization Strategy Tests (refactored for stateless API)
# ============================================================================

add_executable(generic_strategy_tests
    minimizer/core/test_generic_strategy.cpp
)

# Compile device sources directly
target_sources(generic_strategy_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(generic_strategy_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

set_target_properties(generic_strategy_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(generic_strategy_tests PRIVATE cxx_std_17)

# Register generic strategy tests with CTest
gtest_discover_tests(generic_strategy_tests)

# Add to run_tests target
add_custom_target(run_generic_strategy_tests
    COMMAND generic_strategy_tests --gtest_color=yes
    DEPENDS generic_strategy_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running generic minimization strategy tests..."
)

# ============================================================================
# AlgorithmManager Tests (stateful wrapper for strategies)
# ============================================================================

add_executable(algorithm_manager_tests
    minimizer/core/test_algorithm_manager.cpp
)

# Compile device sources directly
target_sources(algorithm_manager_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/algorithm_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(algorithm_manager_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

set_target_properties(algorithm_manager_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(algorithm_manager_tests PRIVATE cxx_std_17)

# Register algorithm manager tests with CTest
gtest_discover_tests(algorithm_manager_tests)

# Add to run_tests target
add_custom_target(run_algorithm_manager_tests
    COMMAND algorithm_manager_tests --gtest_color=yes
    DEPENDS algorithm_manager_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AlgorithmManager tests..."
)

# ============================================================================
# Strategy Consistency Tests (CUDA vs CPU)
# ============================================================================

add_executable(strategy_consistency_tests
    minimizer/core/test_strategy_consistency.cpp
)

# Compile device sources directly
target_sources(strategy_consistency_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(strategy_consistency_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

set_target_properties(strategy_consistency_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(strategy_consistency_tests PRIVATE cxx_std_17)

# Register strategy consistency tests with CTest
gtest_discover_tests(strategy_consistency_tests)

# Add to run_tests target
add_custom_target(run_strategy_consistency_tests
    COMMAND strategy_consistency_tests --gtest_color=yes
    DEPENDS strategy_consistency_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running strategy consistency tests (CUDA vs CPU)..."
)

# ============================================================================
# CheckpointManager Tests (orchestration layer)
# ============================================================================

add_executable(checkpoint_manager_tests
    minimizer/orchestration/test_checkpoint_manager.cpp
)

# Link to management library (which includes checkpoint_manager)
target_link_libraries(checkpoint_manager_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    minimizer_management
    stdc++fs  # For filesystem support
)

target_compile_features(checkpoint_manager_tests PRIVATE cxx_std_17)

# Register checkpoint manager tests with CTest
gtest_discover_tests(checkpoint_manager_tests)

# Add to run_tests target
add_custom_target(run_checkpoint_manager_tests
    COMMAND checkpoint_manager_tests --gtest_color=yes
    DEPENDS checkpoint_manager_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CheckpointManager tests..."
)

# ============================================================================
# Config System Tests (ConfigBuilder, ConfigLoader, etc.)
# ============================================================================

add_executable(config_tests
    test_config.cpp
)

target_link_libraries(config_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    minimizer_config
    yaml-cpp
)

target_compile_features(config_tests PRIVATE cxx_std_17)

# Register config tests with CTest
gtest_discover_tests(config_tests)

# Add to run_tests target
add_custom_target(run_config_tests
    COMMAND config_tests --gtest_color=yes
    DEPENDS config_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running config system tests..."
)

# ============================================================================
# Orchestration Layer Tests - ConcurrentQueue (header-only, no device dependencies)
# ============================================================================

add_executable(concurrent_queue_tests
    orchestration/test_concurrent_queue.cpp
)

target_link_libraries(concurrent_queue_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads  # For std::thread, std::mutex, etc.
)

target_compile_features(concurrent_queue_tests PRIVATE cxx_std_17)

# Register concurrent queue tests with CTest
gtest_discover_tests(concurrent_queue_tests)

# Add to run_tests target
add_custom_target(run_concurrent_queue_tests
    COMMAND concurrent_queue_tests --gtest_color=yes
    DEPENDS concurrent_queue_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running ConcurrentQueue tests..."
)

# ============================================================================
# Orchestration Layer Tests - ResultCollector (header-only, no device dependencies)
# ============================================================================

add_executable(result_collector_tests
    orchestration/test_result_collector.cpp
)

target_link_libraries(result_collector_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads  # For std::thread, std::mutex, etc.
)

target_compile_features(result_collector_tests PRIVATE cxx_std_17)

# Register result collector tests with CTest
gtest_discover_tests(result_collector_tests)

# Add to run_tests target
add_custom_target(run_result_collector_tests
    COMMAND result_collector_tests --gtest_color=yes
    DEPENDS result_collector_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running ResultCollector tests..."
)

# ============================================================================
# ============================================================================
# Orchestration Layer Tests - DevicePool (requires compute backends)
# ============================================================================

add_executable(device_pool_tests
    orchestration/test_device_pool.cpp
)

# Compile device sources directly
target_sources(device_pool_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/device_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/device_registry.cpp
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(device_pool_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
    yaml-cpp  # Required for ResourceMonitor dependency
)

set_target_properties(device_pool_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(device_pool_tests PRIVATE cxx_std_17)

# Register device pool tests with CTest
gtest_discover_tests(device_pool_tests)

# Add to run_tests target
add_custom_target(run_device_pool_tests
    COMMAND device_pool_tests --gtest_color=yes
    DEPENDS device_pool_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DevicePool tests..."
)

# ============================================================================
# Orchestration Layer Tests - RunOrchestrator (requires all minimizer components)
# ============================================================================

add_executable(run_orchestrator_tests
    orchestration/test_run_orchestrator.cpp
)

# Compile all required sources
target_sources(run_orchestrator_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/run_orchestrator.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/algorithm_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/strategy_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_predictor.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/conditions_checker.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/max_iterations_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/convergence_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/numerical_instability_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/prediction_check_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/stop_reason.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/prediction/exponential_fitting_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/prediction/linear_extrapolation_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/management/checkpoint_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/utilities/serializer/serializer.cpp
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(run_orchestrator_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
    yaml-cpp
    Threads::Threads
)

# Add nlohmann/json include directory for CheckpointManager
target_include_directories(run_orchestrator_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/external
)

set_target_properties(run_orchestrator_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(run_orchestrator_tests PRIVATE cxx_std_17)

# Register run orchestrator tests with CTest
gtest_discover_tests(run_orchestrator_tests)

# Add to run_tests target
add_custom_target(run_run_orchestrator_tests
    COMMAND run_orchestrator_tests --gtest_color=yes
    DEPENDS run_orchestrator_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running RunOrchestrator tests..."
)

# ============================================================================
# Orchestration Layer Tests - WorkerThreadPool (requires all orchestration components)
# ============================================================================

add_executable(worker_pool_tests
    orchestration/test_worker_pool.cpp
)

# Compile all required sources
target_sources(worker_pool_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/worker_thread_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/run_orchestrator.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/device_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/device_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/algorithm_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/strategy_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_predictor.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/conditions_checker.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/max_iterations_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/convergence_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/numerical_instability_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/prediction_check_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/stop_reason.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/prediction/exponential_fitting_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/prediction/linear_extrapolation_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/management/checkpoint_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/utilities/serializer/serializer.cpp
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(worker_pool_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
    yaml-cpp
    Threads::Threads
)

# Add nlohmann/json include directory for CheckpointManager
target_include_directories(worker_pool_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/external
)

set_target_properties(worker_pool_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(worker_pool_tests PRIVATE cxx_std_17)

# Register worker pool tests with CTest
gtest_discover_tests(worker_pool_tests)

# Add to run_tests target
add_custom_target(run_worker_pool_tests
    COMMAND worker_pool_tests --gtest_color=yes
    DEPENDS worker_pool_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running WorkerThreadPool tests..."
)

# ============================================================================
# Orchestration Layer Tests - MinimizerOrchestrator (top-level orchestration)
# ============================================================================

add_executable(minimizer_orchestrator_tests
    orchestration/test_minimizer_orchestrator.cpp
)

# Compile all required sources
target_sources(minimizer_orchestrator_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/minimizer_orchestrator.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/worker_thread_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/run_orchestrator.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/device_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/orchestration/device_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/algorithm_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/strategy_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/state/entropy_predictor.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/conditions_checker.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/max_iterations_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/convergence_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/numerical_instability_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/prediction_check_condition.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/stopping/stop_reason.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/prediction/exponential_fitting_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/prediction/linear_extrapolation_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/management/checkpoint_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/utilities/serializer/serializer.cpp
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(minimizer_orchestrator_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
    yaml-cpp
    Threads::Threads
)

# Add nlohmann/json include directory for CheckpointManager
target_include_directories(minimizer_orchestrator_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/external
)

set_target_properties(minimizer_orchestrator_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(minimizer_orchestrator_tests PRIVATE cxx_std_17)

# Register minimizer orchestrator tests with CTest
gtest_discover_tests(minimizer_orchestrator_tests)

# Add to run_tests target
add_custom_target(run_minimizer_orchestrator_tests
    COMMAND minimizer_orchestrator_tests --gtest_color=yes
    DEPENDS minimizer_orchestrator_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running MinimizerOrchestrator tests..."
)

# ============================================================================
# StrategyFactory Tests
# ============================================================================

add_executable(strategy_factory_tests
    test_strategy_factory.cpp
)

# Compile all required sources
target_sources(strategy_factory_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/strategy_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/generic_minimization_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/minimizer/algorithm/cuda_minimization_strategy.cu
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
    ${CMAKE_SOURCE_DIR}/src/compute/device/DeviceFactory.cu
)

target_link_libraries(strategy_factory_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

set_target_properties(strategy_factory_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(strategy_factory_tests PRIVATE cxx_std_17)

# Register strategy factory tests with CTest
gtest_discover_tests(strategy_factory_tests)

# Add to run_tests target
add_custom_target(run_strategy_factory_tests
    COMMAND strategy_factory_tests --gtest_color=yes
    DEPENDS strategy_factory_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running StrategyFactory tests..."
)
