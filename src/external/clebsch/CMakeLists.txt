# src/external/clebsch/CMakeLists.txt
# Clebsch-Gordan Coefficients Library
# External source: https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ClebschGordan.cpp
# Original author: Arne Alex

add_library(clebsch STATIC sud_cg.cpp)

# C++ standard requirement
target_compile_features(clebsch PUBLIC cxx_std_17)

# Include the current directory for headers
target_include_directories(clebsch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Find LAPACK - it's optional but recommended
find_package(LAPACK)

if(LAPACK_FOUND)
    # Link LAPACK
    target_link_libraries(clebsch PUBLIC LAPACK::LAPACK)
    
    # Try to detect and set backend definition
    # This is used in lapack_interface.h to include the right headers
    if(APPLE AND "${LAPACK_LIBRARIES}" MATCHES "Accelerate")
        target_compile_definitions(clebsch PUBLIC LAPACK_ACCELERATE ACCELERATE_NEW_LAPACK)
        message(STATUS "Clebsch: Using Apple Accelerate framework")
    elseif("${LAPACK_LIBRARIES}" MATCHES "lapacke|openblas")
        target_compile_definitions(clebsch PUBLIC LAPACK_OPENBLAS)
        message(STATUS "Clebsch: Using OpenBLAS (with lapacke)")
    elseif("${LAPACK_LIBRARIES}" MATCHES "mkl")
        target_compile_definitions(clebsch PUBLIC LAPACK_MKL)
        message(STATUS "Clebsch: Using Intel MKL")
    else()
        # Generic system LAPACK - use Fortran interface only
        message(STATUS "Clebsch: Using system LAPACK (Fortran interface)")
    endif()
    
    message(STATUS "Clebsch: LAPACK libraries: ${LAPACK_LIBRARIES}")
else()
    message(STATUS "Clebsch: LAPACK not found - will use Fortran interface declarations")
endif()