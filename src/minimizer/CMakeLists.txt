# src/minimizer/CMakeLists.txt
# Main minimizer module with all submodules

# Add all minimizer submodules
add_subdirectory(config)
add_subdirectory(core)
add_subdirectory(utilities)
add_subdirectory(analysis)
add_subdirectory(algorithm)

# Create a top-level minimizer library that aggregates everything
add_library(minimizer INTERFACE)

target_link_libraries(minimizer INTERFACE
    minimizer_config
    minimizer_core
    minimizer_utilities
    minimizer_analysis
    minimizer_algorithm
)

message(STATUS "Minimizer module configured: core minimizer with config, utilities, analysis, and algorithm")
