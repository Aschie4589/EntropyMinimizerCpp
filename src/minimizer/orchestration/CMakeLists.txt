# Orchestration layer for work-stealing and device management

add_library(minimizer_orchestration
    device_pool.cpp
)

target_include_directories(minimizer_orchestration PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(minimizer_orchestration PUBLIC
    yaml-cpp
)

target_compile_features(minimizer_orchestration PUBLIC cxx_std_17)
