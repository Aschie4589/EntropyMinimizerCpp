#include "utilities/messaging/printer.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace utils {

Printer::Printer(int min_level, bool color)
    : uuid_(generate_uuid_v4()), min_level_(min_level), color_enabled_(color) {}

void Printer::send(const std::string& msg, int level) {
    if (shouldSend(level)) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::ostringstream timestamp;
        timestamp << std::put_time(std::localtime(&time_t), "%Y-%m-%d:%H:%M:%S")
                  << "." << std::setfill('0') << std::setw(3) << ms.count();
        
        std::string level_str;
        switch(level) {
            case 0: level_str = "INFO"; break;
            case 1: level_str = "WARN"; break;
            case 2: level_str = "ERROR"; break;
            default: level_str = "L" + std::to_string(level); break;
        }
        
        std::cout << "[" << timestamp.str() << "] " << level_str << ": " << msg << std::endl;
    }
}    

bool Printer::shouldSend(int level) const {
    return level >= min_level_;
}


std::string Printer::getLevelColor(int level) const {
    // Example color coding
    if (level >= 3) return "\033[1;31m";  // Red for errors
    if (level >= 2) return "\033[1;33m";  // Yellow for warnings
    return "\033[1;37m";                   // White for info
}

} // namespace utils
