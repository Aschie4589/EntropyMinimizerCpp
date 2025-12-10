# Minimizer Stopping Conditions Library
# Stopping condition implementations for minimization runs

add_library(minimizer_stopping
    stop_reason.cpp
    max_iterations_condition.cpp
    convergence_condition.cpp
    numerical_instability_condition.cpp
    prediction_check_condition.cpp
    conditions_checker.cpp
)

target_include_directories(minimizer_stopping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(minimizer_stopping PUBLIC cxx_std_17)
