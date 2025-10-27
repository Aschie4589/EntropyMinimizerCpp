# src/utilities/serializer/CMakeLists.txt
# Serialization utilities for vectors and Kraus operators

add_library(serializer STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/serializer.cpp
)

target_compile_features(serializer PUBLIC cxx_std_17)

target_include_directories(serializer
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/external
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(serializer PUBLIC
    nlohmann_json
)

message(STATUS "Serializer module configured: serializer library")
