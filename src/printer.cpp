#include <chrono>
#include <ctime>

#include "common_includes.h"

#include "printer.h"
#include "config.h"

namespace fs = std::filesystem;

Printer::Printer(std::string uuidStr)
{
    // Copy the UUID over
    int l = std::min(uuidStr.length(), static_cast<std::size_t>(36)); // the cast is necessary bc min only works with two of the same type objects
    for (int i=0; i<l; i++){
        uuid[i] = uuidStr.at(i);
    }
    uuid[36] = '\0';

}

Printer::Printer() : Printer("123e4567-e89b-12d3-a456-426614174000"){
    // Handle the missing UUID, then call the other logger!
    // TODO: IMPLEMENT!!!
    std::cout << "Printer was called without UUID. Since UUID creation has not been implemented, will use 123e4567-e89b-12d3-a456-426614174000 as a default!" << std::endl;
}


int Printer::printMessage(std::string message, int log_level){
    std::cout << getPrintContext(log_level)<<message << std::endl;
    return 0;
}

int Printer::printMessage(std::string message){
    // If no level is specified, assume it is just info.
    return printMessage(message, LOG_LEVEL_INFO);
}

std::string Printer::getCurrentTimeString() {
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

std::string Printer::getPrintContext(int log_level){
    // Create a stringstream to add all the relevant context info
    std::ostringstream oss;
    if (PRINTER_CONTEXT_TIMESTAMP){
        oss << "[ "<<getCurrentTimeString()<<" ] - ";
    }
    if (PRINTER_CONTEXT_UUID) {
        oss << "Logger "<<uuid<<" - ";
    }
    if (PRINTER_CONTEXT_LOG_LEVEL) {
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

Printer::~Printer()
{
}
