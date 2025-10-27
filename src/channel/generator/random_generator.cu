#include "channel/generator/random_generator.h"
#include "channel/utilities/generate_haar_unitary.h"

#include <cuda.h>
#include <cuComplex.h>
#include <cublas_v2.h>


int RandomGenerator::generate(std::vector<std::complex<double>>* kraus) {
    /*
    Returns a vector of Kraus operators, each represented as a flat vector in column-major order
    Updates the input pointer kraus in place! Requires that the pointer is not null.
    */
    // generate the Kraus operators, DIRECTLY ON DEVICE
    // Allocate kraus operators
    cuDoubleComplex* kraus_operators;
    cudaError_t errmalloc = cudaMalloc(&kraus_operators, config.kraus_number * config.kraus_in_dimension * config.kraus_out_dimension * sizeof(cuDoubleComplex));
    if (errmalloc != cudaSuccess) {
        config.message_handler->message("Error allocating memory for Kraus operators: " + std::string(cudaGetErrorString(errmalloc)));
        return 1;
    }
    config.message_handler->message("Successfully allocated " + std::to_string(config.kraus_number * config.kraus_in_dimension * config.kraus_out_dimension * sizeof(cuDoubleComplex)) + " bytes or " + std::to_string(config.kraus_number * config.kraus_in_dimension * config.kraus_out_dimension * sizeof(cuDoubleComplex)/1024.0/1024/1024) + " GiB for Kraus operators on device.");
    // Generate Haar random unitaries
    cudaError_t err = generateHaarRandomUnitaries(kraus_operators, config.kraus_in_dimension, config.kraus_number, 32);
    if (err != cudaSuccess) {
        config.message_handler->message("Error generating Haar random unitaries: " + std::string(cudaGetErrorString(err)));
        return 1;
    }

    // Rescale the unitaries using cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1/sqrt(config.kraus_number);
    cublasZdscal(handle, config.kraus_number*config.kraus_in_dimension * config.kraus_out_dimension, &alpha, kraus_operators, 1);

    // Destroy the handle
    cublasDestroy(handle);

    // Copy the kraus ops over to device for saving
    cudaError_t errcopy = cudaMemcpy(kraus->data(), kraus_operators, config.kraus_number * config.kraus_in_dimension * config.kraus_out_dimension * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (errcopy != cudaSuccess) {
        config.message_handler->message("Error copying Kraus operators from device to host: " + std::string(cudaGetErrorString(errcopy)));
        return 1;
    }
    config.message_handler->message("Successfully copied " + std::to_string(config.kraus_number * config.kraus_in_dimension * config.kraus_out_dimension * sizeof(cuDoubleComplex)) + " bytes or " + std::to_string(config.kraus_number * config.kraus_in_dimension * config.kraus_out_dimension * sizeof(cuDoubleComplex)/1024.0/1024/1024) + " GiB of Kraus operators from device to host.");
    // Free the device memory
    cudaError_t errfree = cudaFree(kraus_operators);
    if (errfree != cudaSuccess) {
        config.message_handler->message("Error freeing memory for Kraus operators: " + std::string(cudaGetErrorString(errfree)));
        return 1;
    }
    return 0;

}

