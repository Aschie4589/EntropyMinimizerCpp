# Orchestration layer for work-stealing and device management

add_library(minimizer_orchestration
    device_pool.cpp
    run_orchestrator.cpp
    worker_thread_pool.cpp
    minimizer_orchestrator.cpp
    device_registry.cpp
)

target_include_directories(minimizer_orchestration PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(minimizer_orchestration PUBLIC
    yaml-cpp
    CUDA::cudart
    minimizer_algorithm
    minimizer_state
    minimizer_stopping
    minimizer_prediction
    minimizer_management
)

target_compile_features(minimizer_orchestration PUBLIC cxx_std_17)
