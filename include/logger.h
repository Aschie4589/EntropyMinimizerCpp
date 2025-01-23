#ifndef LOGGER_H
#define LOGGER_H

#include "common_includes.h"
#include "config.h"

namespace fs = std::filesystem;
class Logger
{
private:
    char uuid[37]; // UUID4 for the logger. Note that a UUID is 36 characters long, but strings terminate with \0 in c++.
    int log_level;

    // Utilities
    std::string getCurrentTimeString(bool millis=true);         //Returns the current time, formatted as a string (formatting is)
    std::string getLogContext(int log_level);

    /* data */
public:
    Logger(std::string filename, std::string uuidStr, int log_level=LOG_LEVEL_INFO);
    Logger(std::string filename, int log_level=LOG_LEVEL_INFO);
    Logger(int log_level=LOG_LEVEL_INFO);
    std::string log_file;


    int logMessage(const std::string& message);
    int logMessage(const std::string& message, int log_level);

    ~Logger();
};




#endif