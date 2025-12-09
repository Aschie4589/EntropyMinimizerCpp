# Minimizer State Library
# State management and entropy tracking

add_library(minimizer_state
    entropy_manager.cpp
    entropy_predictor.cpp
)

target_include_directories(minimizer_state PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(minimizer_state PUBLIC
    minimizer_prediction
)

target_compile_features(minimizer_state PUBLIC cxx_std_17)
