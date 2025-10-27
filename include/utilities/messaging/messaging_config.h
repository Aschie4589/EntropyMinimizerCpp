#ifndef MESSAGING_CONFIG_H
#define MESSAGING_CONFIG_H

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
File save parameters
*/
#define SAVE_DIRECTORY "save"           // Parent folder 
#define LOG_DIRECTORY "logs"            // Subfolder for saving logs

/*
Misc
*/
#define PRINT_PRECISION 15  // How many digits to print in print statements


#endif