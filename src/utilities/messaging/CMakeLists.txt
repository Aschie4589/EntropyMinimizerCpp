# src/utilities/messaging/CMakeLists.txt
# Messaging utilities: logging, printing, and message handling

add_library(messaging STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/message_handler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/printer.cpp
)

target_compile_features(messaging PUBLIC cxx_std_17)

target_include_directories(messaging
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(messaging PUBLIC
    uuid
)

message(STATUS "Messaging module configured: messaging library")