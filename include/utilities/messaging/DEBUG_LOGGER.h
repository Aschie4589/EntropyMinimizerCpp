#ifndef DEBUG_LOGGER_H_
#define DEBUG_LOGGER_H_


// Macro to log to file
// Use: DEBUG_LOG("Message to log", filename);
#include <fstream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <string>

//#define DEBUG_LOGGER_ENABLED 

#ifdef DEBUG_LOGGER_ENABLED 
#define DEBUG_LOG(msg, filename) do { \
    std::ofstream log_file((filename), std::ios::app); \
    auto now = std::chrono::system_clock::now(); \
    auto time_t_now = std::chrono::system_clock::to_time_t(now); \
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( \
        now.time_since_epoch()) % 1000; \
    log_file << "[" << std::put_time(std::localtime(&time_t_now), "%H:%M:%S") \
             << "." << std::setfill('0') << std::setw(3) << ms.count() \
             << "] " << (msg) << std::endl; \
    log_file.close(); \
} while(0)
#else
#define DEBUG_LOG(msg, filename) do { } while(0)
#endif

#endif // DEBUG_LOGGER_H_