#include <cuda.h>
#include <cuComplex.h>
#include <cublas_v2.h>

// Headers
#include "channel/generator/random_generator.h"

// Utilities
#include "channel/utilities/generate_haar_unitary.h"

// RAII Cuda memory management
#include "utilities/cuda/cuda_device_memory.h"
#include "utilities/cuda/cuda_blas_handle.h"
#include "utilities/cuda/error_handling.h"

int RandomGenerator::generate(std::vector<std::complex<double>>* kraus) {
    /*
    Returns a vector of Kraus operators, each represented as a flat vector in column-major order
    Updates the input pointer kraus in place! Requires that the pointer is not null.
    */
    
    try {
        const size_t kraus_size = config.kraus_number * 
                                  config.kraus_in_dimension * 
                                  config.kraus_out_dimension;
        const size_t size_bytes = kraus_size * sizeof(cuDoubleComplex);
        const double size_gib = size_bytes / (1024.0 * 1024.0 * 1024.0);
        
        CudaDeviceMemory<cuDoubleComplex> d_kraus_operators(kraus_size);
        
        msg_handler_.info("Successfully allocated " + 
            std::to_string(size_bytes) + " bytes (" + 
            std::to_string(size_gib) + " GiB) for Kraus operators on device.");
        
        CUDA_CHECK(generateHaarRandomUnitaries(
            d_kraus_operators.get(), 
            config.kraus_in_dimension, 
            config.kraus_number, 
            32,
            &msg_handler_  // Pass message handler for debug logging
        ));
        
        CudaBlasHandle blas_handle;
        
        double alpha = 1.0 / sqrt(config.kraus_number);
        CUBLAS_CHECK(cublasZdscal(
            blas_handle.get(), 
            kraus_size, 
            &alpha, 
            d_kraus_operators.get(), 
            1
        ));
        
        d_kraus_operators.copyToHost(
            reinterpret_cast<cuDoubleComplex*>(kraus->data()), 
            kraus_size
        );
        
        msg_handler_.info("Successfully copied " + 
            std::to_string(size_bytes) + " bytes (" + 
            std::to_string(size_gib) + " GiB) of Kraus operators from device to host.");
        
        // d_kraus_operators freed here
        // blas_handle destroyed here
        
        return 0;
        
    } catch (const std::runtime_error& e) {
        msg_handler_.error("Error in RandomGenerator::generate: " + std::string(e.what()));
        return 1;
    } catch (const std::exception& e) {
        msg_handler_.error("Unexpected error in RandomGenerator::generate: " + std::string(e.what()));
        return 1;
    }
}

