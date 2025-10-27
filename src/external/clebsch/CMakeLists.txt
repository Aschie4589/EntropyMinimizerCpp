# src/external/clebsch/CMakeLists.txt
# Clebsch-Gordan Coefficients Library
# External source: https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ClebschGordan.cpp
# Original author: Arne Alex

add_library(clebsch STATIC sud_cg.cpp)

# C++ standard requirement
target_compile_features(clebsch PUBLIC cxx_std_17)

# Include the current directory for headers
target_include_directories(clebsch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Use the LAPACK configuration from root CMakeLists
target_include_directories(clebsch PUBLIC ${LAPACK_INCLUDE_DIR})
target_link_libraries(clebsch PUBLIC ${LAPACK_LIBRARIES})

# Set compile definitions based on LAPACK backend used in root
if(LAPACK_OPENBLAS)
    target_compile_definitions(clebsch PUBLIC LAPACK_OPENBLAS)
    message(STATUS "Clebsch: Using OpenBLAS backend")
else()
    # Fallback to system LAPACK (Fortran interface only)
    message(STATUS "Clebsch: Using system LAPACK (Fortran interface)")
endif()