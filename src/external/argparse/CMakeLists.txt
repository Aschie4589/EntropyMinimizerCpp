# argparse - header-only library
# A single-header C++ argument parser library
# URL: https://github.com/p-ranav/argparse

add_library(argparse INTERFACE)

# Header-only interface
target_include_directories(argparse INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Enforce C++17 for clients
target_compile_features(argparse INTERFACE cxx_std_17)

message(STATUS "argparse configured (header-only)")
