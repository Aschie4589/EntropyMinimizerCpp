# src/minimizer/analysis/CMakeLists.txt
# Entropy estimation and convergence analysis

# Find LAPACK for linear algebra operations
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(minimizer_analysis STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/entropy_estimator.cpp
)

target_compile_features(minimizer_analysis PUBLIC cxx_std_17)

target_include_directories(minimizer_analysis
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        $ENV{HOME}/.local/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(minimizer_analysis PUBLIC
    minimizer_config
    BLAS::BLAS
    LAPACK::LAPACK
)

message(STATUS "Minimizer analysis module configured: minimizer_analysis library")
