# Minimizer Prediction Library
# Prediction strategies for entropy convergence

add_library(minimizer_prediction
    exponential_fitting_strategy.cpp
    linear_extrapolation_strategy.cpp
)

target_include_directories(minimizer_prediction PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(minimizer_prediction PUBLIC cxx_std_17)
