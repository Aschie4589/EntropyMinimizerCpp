/*
CONFIG

Here all the configuration parameters for the minimizer are stored.
These are expanded at compile time, so they will be hard-coded in the program.
*/

#ifndef CONFIG_H
#define CONFIG_H

/* 
EntropyMinimizer parameters
*/

// Define version number
#define VERSION "0.1"


// Default parameters that can be changed in MinimizerConfig
#define DEFAULT_MINIMIZER_MAX_ITERATIONS 500000             // How many iterations of the algorithm to run before giving up. Has to be less than uint32_t range
#define DEFAULT_MINIMIZER_LOG false                         // Should the minimizer write messages to a log file?
#define DEFAULT_MINIMIZER_LOG_PREFIX "log"
#define DEFAULT_MINIMIZER_PRINT true                        // Should the minimizer print messages to the console?
#define DEFAULT_MINIMIZER_EPSILON 1.0f/1000                 // What is the default epsilon by which to perturb the channel to ensure full rank?
#define DEFAULT_MINIMIZER_USE_MOE_PREDICTION true           // Should the minimizer use the prediction of the final MOE to stop the algorithm?
#define DEFAULT_MINIMIZER_MOE_PREDICTION_TOLERANCE 1e-5     // What is the tolerance for the MOE prediction?
#define DEFAULT_MINIMIZER_MINIMIZATION_ATTEMPTS 100         // How many times to run the minimization algorithm before giving up

#define DEFAULT_MINIMIZER_CHECKPOINT_INTERVAL 100           // How often to save the state of the minimizer
#define DEFAULT_MINIMIZER_CHECKPOINT_FILE "checkpoint.dat"      // What is the default name of the checkpoint file


// These other parameters that are just baked in at compile
#define CONVERGENCE_TOLERANCE 1e-15     // When running the algorithm, if the improvement is below this threshold value for CONVERGENCE_ITERS iterations, 
#define CONVERGENCE_ITERS 20            // How many iterations to average over to check for convergence
#define RSQUARED_THRESHOLD 0.999        // What is the threshold for the R^2 value of the linear fit to be considered good enough

/*
Entropy estimator parameters
*/
#define ENTROPY_ESTIMATOR_MAX_WINDOW_SIZE 500
#define ENTROPY_ESTIMATOR_DEFAULT_WINDOW_SIZE 200


/*
File save parameters
*/
#define SAVE_DIRECTORY "save"           // Parent folder 
#define LOG_DIRECTORY "logs"            // Subfolder for saving logs
#define VECTORS_DIRECTORY "checkpoints"     // Subfolder for saving vectors
#define KRAUS_DIRECTORY "kraus"         // Subfolder for saving kraus operators

/*
LOGGING configuration. These are baked in.
*/
#define LOGGER_CONTEXT_TIMESTAMP true       // Whether to include timestamp in the log line or not
#define LOGGER_CONTEXT_UUID false            // Whether to include the logger UUID in the log line
#define LOGGER_CONTEXT_LOG_LEVEL true       // Whether to include the log level (INFO, WARNING, ERROR)
#define PRINTER_CONTEXT_TIMESTAMP true       // Whether to include timestamp in the printed line or not
#define PRINTER_CONTEXT_UUID false            // Whether to include the logger UUID in the printed line
#define PRINTER_CONTEXT_LOG_LEVEL false            // Whether to include the log level (INFO, WARNING, ERROR)
#define LOG_LEVEL_INFO 0
#define LOG_LEVEL_WARNING 1
#define LOG_LEVEL_ERROR 2
#define LOGGER_TIME_FORMAT "%Y-%m-%d %H:%M:%S"
#define LOGGER_APPEND_MILLIS true           // This appends ".mmm" at the end of the timestamp string, however it may be formatted!


/*
Misc
*/
#define PRINT_PRECISION 15  // How many digits to print in print statements

#endif