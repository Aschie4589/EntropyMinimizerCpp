# Minimizer Management Library
# High-level orchestration and checkpoint management

add_library(minimizer_management STATIC
    checkpoint_manager.cpp
)

target_include_directories(minimizer_management PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(minimizer_management PUBLIC
    minimizer_config
    serializer
    uuid
    nlohmann_json  # For transitive include of nlohmann/json.hpp
)

set_target_properties(minimizer_management PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)
