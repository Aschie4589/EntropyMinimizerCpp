#ifndef PRINTER_H
#define PRINTER_H

#include "common_includes.h"
#include "config.h"

namespace fs = std::filesystem;
class Printer
{
private:
    char uuid[37]; // UUID4 for the printer. Note that a UUID is 36 characters long, but strings terminate with \0 in c++.

    // Utilities
    std::string getCurrentTimeString();         //Returns the current time, formatted as a string (formatting is)
    std::string getPrintContext(int log_level);

    /* data */
public:
    Printer(std::string uuidStr);
    Printer();

    int printMessage(std::string message);
    int printMessage(std::string message, int log_level);

    ~Printer();
};




#endif