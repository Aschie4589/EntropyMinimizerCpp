# src/utilities/uuid/CMakeLists.txt
# UUID generation utility

add_library(uuid STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/uuid.cpp
)

target_compile_features(uuid PUBLIC cxx_std_17)

target_include_directories(uuid
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(uuid PUBLIC
    # Standard library only - no external dependencies
)

message(STATUS "UUID module configured: uuid library")
