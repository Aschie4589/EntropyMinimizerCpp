# src/minimizer/algorithm/CMakeLists.txt
# High-level entropy minimization algorithm orchestration

add_library(minimizer_algorithm STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/entropy_minimizer.cu
)

set_target_properties(minimizer_algorithm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_compile_features(minimizer_algorithm PUBLIC cxx_std_17)

target_include_directories(minimizer_algorithm
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/external
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(minimizer_algorithm PUBLIC
    minimizer_core
    minimizer_config
    minimizer_analysis
    messaging
    serializer
    uuid
    CUDA::cudart
)

message(STATUS "Minimizer algorithm module configured: minimizer_algorithm library")
