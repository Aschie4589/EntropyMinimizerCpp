/*
CONFIG

Here all the configuration parameters for the minimizer are stored.
These are expanded at compile time, so they will be hard-coded in the program.
*/

#ifndef CONFIG_H
#define CONFIG_H

/* 
Minimizer parameters
*/
#define EPS 1.0f/1000                      // The default value of epsilon that is passed to the minimizer.

#define MAX_ITERATIONS 500000           // How many iterations of the algorithm to run before giving up. Has to be less than uint32_t range

#define CONVERGENCE_TOLERANCE 1e-15     // When running the algorithm, if the improvement is below this threshold value for CONVERGENCE_ITERS iterations, 
#define CONVERGENCE_ITERS 20            // 

/*
The Minimizer class provides a few methods for printing vectors and matrices (which are not too advanced!)
Here you can change their behavior.
*/
#define PRINT_PRECISION 15  // How many digits to print in print statements


/*
File save parameters
*/
#define SAVE_DIRECTORY "save"           // Parent folder 
#define LOG_DIRECTORY "logs"            // Subfolder for saving logs
#define VECTORS_DIRECTORY "checkpoints"     // Subfolder for saving vectors
#define KRAUS_DIRECTORY "kraus"         // Subfolder for saving kraus operators

/*
LOGGING
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

#endif