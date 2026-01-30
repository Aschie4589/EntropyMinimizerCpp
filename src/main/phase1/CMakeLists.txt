# src/main/phase1/CMakeLists.txt
# Phase 1 application configuration
# Note: The phase1 executable is built in the root CMakeLists.txt
# This file documents the phase1-specific sources and headers

message(STATUS "Phase 1 application sources ready")

# Phase 1 includes its own tensor_entropy implementation
# which is compiled directly into the phase1 executable
# See root CMakeLists.txt for phase1 executable configuration
