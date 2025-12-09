# Config library
add_library(minimizer_config INTERFACE)

target_include_directories(minimizer_config INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

# Config depends on yaml-cpp (already in project via external/)
target_link_libraries(minimizer_config INTERFACE
    yaml-cpp
)

# Config is header-only, so no source files to compile
# All functionality is in headers
