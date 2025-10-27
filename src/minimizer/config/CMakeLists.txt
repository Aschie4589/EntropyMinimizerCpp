# src/minimizer/config/CMakeLists.txt
# Minimizer configuration management

add_library(minimizer_config STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/entropy_config.cpp
)

target_compile_features(minimizer_config PUBLIC cxx_std_17)

target_include_directories(minimizer_config
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

message(STATUS "Minimizer config module configured: minimizer_config library")
