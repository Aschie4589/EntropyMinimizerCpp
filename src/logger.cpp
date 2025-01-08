#include <chrono>
#include <ctime>

#include "common_includes.h"

#include "logger.h"
#include "config.h"

namespace fs = std::filesystem;

Logger::Logger(std::string filename, std::string uuidStr, int log_level) : log_file(filename)
{
    // Copy the UUID over
    int l = std::min(uuidStr.length(), static_cast<std::size_t>(36)); // the cast is necessary bc min only works with two of the same type objects
    for (int i=0; i<l; i++){
        uuid[i] = uuidStr.at(i);
    }
    uuid[36] = '\0';

    // Set the log level. Log level means that all messages with that priority, or higher, are printed, while those lower are discarded.
    this->log_level = log_level;
    // Ensure that the directory to save the log file exists
    fs::create_directories(fs::current_path()/fs::path(SAVE_DIRECTORY) / fs::path(LOG_DIRECTORY));


}

Logger::Logger(std::string filename, int log_level) : Logger(filename, "123e4567-e89b-12d3-a456-426614174000", log_level) {
    // Handle the missing UUID, then call the other logger!
    // TODO: IMPLEMENT!!! Note that I need a function called above...! because that is the only way to "delegate a constructor". 
    std::cout << "Logger was called without UUID. Since UUID creation has not been implemented, will use 123e4567-e89b-12d3-a456-426614174000 as a default!" << std::endl;

}


int Logger::logMessage(const std::string& message, int log_level){
    
    // Check the log level
    if (log_level >= this->log_level){
        // Open the file for 
        std::ofstream out_file(fs::current_path()/fs::path(SAVE_DIRECTORY) / fs::path(LOG_DIRECTORY) / fs::path(log_file), std::ios::app); // Open log file. std::ios::app tells the stream to append to the end of the file and not overwrite.
        if (out_file.is_open()) {
            // Write the line to the file
            out_file << getLogContext(log_level)<<message << std::endl;
            
            // Optionally, explicitly close the file (though it's closed automatically when outFile goes out of scope)
            out_file.close();
        } else {
            return 1;
        }
    }
    return 0;
}

int Logger::logMessage(const std::string& message){
    // If no level is specified, assume it is just info.
    return logMessage(message, LOG_LEVEL_INFO);
}

std::string Logger::getCurrentTimeString() {
    // Get the current time as a time_point
    auto now = std::chrono::system_clock::now();
    
    // Convert to time_t (the type used for time)
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    // Format the time as a string
    std::tm tm = *std::localtime(&now_c);
    
    // Create a stringstream to format the time in a custom format
    std::ostringstream oss;
    oss << std::put_time(&tm, LOGGER_TIME_FORMAT); 

    // Append milliseconds if needed
    if (LOGGER_APPEND_MILLIS){
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        oss << "." << std::setw(3) << std::setfill('0') << ms.count();
    }
    return oss.str();
}

std::string Logger::getLogContext(int log_level){
    // Create a stringstream to add all the relevant context info
    std::ostringstream oss;
    if (LOGGER_CONTEXT_TIMESTAMP){
        oss << "[ "<<getCurrentTimeString()<<" ] - ";
    }
    if (LOGGER_CONTEXT_UUID) {
        oss << "Logger "<<uuid<<" - ";
    }
    if (LOGGER_CONTEXT_LOG_LEVEL) {
        if (log_level == LOG_LEVEL_INFO){
            oss << "INFO: ";
        } else if (log_level == LOG_LEVEL_WARNING){
            oss << "WARNING: ";
        } else if (log_level == LOG_LEVEL_ERROR){
            oss << "ERROR: ";
        } else {
            oss << "UNKNOWN LOG LEVEL: ";
        }
    }
    return oss.str();
}

Logger::~Logger()
{
}
