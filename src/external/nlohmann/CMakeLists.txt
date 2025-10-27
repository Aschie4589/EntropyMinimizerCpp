# nlohmann/json - header-only JSON library
# A modern C++11 JSON library with an intuitive and straightforward API
# URL: https://github.com/nlohmann/json

add_library(nlohmann_json INTERFACE)

# Header-only interface
target_include_directories(nlohmann_json INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Enforce C++17 for clients
target_compile_features(nlohmann_json INTERFACE cxx_std_17)

message(STATUS "nlohmann/json configured (header-only)")
