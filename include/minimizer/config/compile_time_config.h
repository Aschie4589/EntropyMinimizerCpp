/*
CONFIG

Here all the configuration parameters for the minimizer are stored.
These are expanded at compile time, so they will be hard-coded in the program.
*/

#ifndef COMPILE_TIME_CONFIG_H
#define COMPILE_TIME_CONFIG_H

/* 
EntropyMinimizer parameters
*/

// Define version number
#define VERSION "0.1"


/*
CUDA Parameters
*/

#define CUDA_STREAMS 32                 // How many CUDA streams to use for parallel execution of non-parallelized operations. Relevant for Haar unitary creation
#define ZGEMM_BATCH_SIZE_CHANNEL 32     // How many matrix multiplications to perform in parallel when applying the channel. This is the batch size for the zgemm operation. Memory usage is proportional to this.
#define MINIMIZER_CUDA_STREAMS 1        // How many CUDA streams to use for parallel execution of minimizer matrix-vector operations.


/*
EntropyConfig default parameters
*/

#define DEFAULT_MINIMIZER_MAX_ITERATIONS 500000             // How many iterations of the algorithm to run before giving up. Has to be less than uint32_t range
#define DEFAULT_MINIMIZER_LOG false                         // Should the minimizer write messages to a log file?
#define DEFAULT_MINIMIZER_LOG_PREFIX "log"
#define DEFAULT_MINIMIZER_PRINT true                        // Should the minimizer print messages to the console?
#define DEFAULT_MINIMIZER_EPSILON 1.0f/1000                 // What is the default epsilon by which to perturb the channel to ensure full rank?
#define DEFAULT_MINIMIZER_USE_MOE_PREDICTION true           // Should the minimizer use the prediction of the final MOE to stop the algorithm?
#define DEFAULT_MINIMIZER_MOE_PREDICTION_TOLERANCE 1e-5     // What is the tolerance for the MOE prediction?
#define DEFAULT_MINIMIZER_MINIMIZATION_ATTEMPTS 100         // How many times to run the minimization algorithm before giving up

#define DEFAULT_MINIMIZER_MOE_DOUBLE_PRECISION true         // Should the MOE be calculated in double precision?
#define DEFAULT_MINIMIZER_MOE_VARIABLE_PRECISION false       // Should the MOE be calculated

#define DEFAULT_MINIMIZER_CHECKPOINT_INTERVAL 100           // How often to save the state of the minimizer
#define DEFAULT_MINIMIZER_CHECKPOINT_FILE "checkpoint.dat"      // What is the default name of the checkpoint file

/*
EntropyMinimizer parameters
*/
#define CONVERGENCE_TOLERANCE 1e-15     // When running the algorithm, if the improvement is below this threshold value for CONVERGENCE_ITERS iterations, 
#define CONVERGENCE_ITERS 20            // How many iterations to average over to check for convergence

/*
Entropy estimator parameters
*/
#define ENTROPY_ESTIMATOR_MAX_WINDOW_SIZE 500
#define ENTROPY_ESTIMATOR_DEFAULT_WINDOW_SIZE 200
#define RSQUARED_THRESHOLD 0.999        // What is the threshold for the R^2 value of the linear fit to be considered good enough


/*
File save parameters
*/
#define SAVE_DIRECTORY "save"           // Parent folder 
#define LOG_DIRECTORY "logs"            // Subfolder for saving logs
#define VECTORS_DIRECTORY "checkpoints"     // Subfolder for saving vectors
#define KRAUS_DIRECTORY "kraus"         // Subfolder for saving kraus operators


/* 
DEFS - These are definitions that are used throughout the program, and shouldn't be changed
*/

/*
Cuda Minimizer
*/
#define CUDA_MINIMIZER_CREATED 0    // The minimizer has been created, but the vector not initialized. The kraus ops are correct. Memory is allocated but not initialized.
#define CUDA_MINIMIZER_STAGE_0 1    // At stage 0, the input vector is initialized and correct. Also kraus is initialized and correct. d_vecs_1 and d_vecs_2 are initialized but empty. d_sv_1 and d_sv_2 are initialized but empty. The entropy is initialized to -1.
#define CUDA_MINIMIZER_STAGE_1 2    // At stage 1, same as stage zero. On top of that, d_vecs_1 and d_sv_1 are filled with the results of the first SVD. It is possible to compute entropy.
#define CUDA_MINIMIZER_STAGE_2 3    // At stage 2, also d_vecs_2 and d_sv_2 are filled with the results of the second SVD. d_vec is updated to the new vector, so the entropy can't be computed for this new vector yet. Computing the entropy will give the old one.




/*
Return codes
*/
#define ENTROPY_MINIMIZER_CONTINUE 0
#define ENTROPY_MINIMIZER_CONVERGED 1
#define ENTROPY_MINIMIZER_MAX_ITERS 2
#define ENTROPY_MINIMIZER_NUMERICAL_INST 3
#define ENTROPY_MINIMIZER_TERMINATED 4
#define ENTROPY_MINIMIZER_MOE_PREDICTION 5


/* MACROS */
// Custom macro to handle cudaMalloc errors
#define CUDA_MALLOC_CHECK(ptr, size, name) \
    do { \
        ptr = nullptr; \
        cudaError_t err = cudaMalloc((void**)&ptr, size); \
        if (err != cudaSuccess) { \
            std::cerr << "Error allocating memory for " << name << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("Failed to allocate memory for " + std::string(name)); \
        } \
    } while(0)

#define CUDA_MEMCPY_CHECK(dest, src, size, kind, name) \
    do { \
        cudaError_t err = cudaMemcpy(dest, src, size, kind); \
        if (err != cudaSuccess) { \
            std::cerr << "Error copying memory for " << name << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("Failed to copy memory for " + std::string(name)); \
        } \
    } while(0)





#endif