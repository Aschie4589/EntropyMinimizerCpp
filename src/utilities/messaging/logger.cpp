#include "utilities/messaging/logger.h"

namespace entropy {

Logger::Logger(const std::string& filename, int min_level)
    : uuid_(generate_uuid_v4()), min_level_(min_level) {
    file_.open(filename, std::ios::app);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open log file: " + filename);
    }
}
Logger::~Logger() {
    if (file_.is_open()) {
        file_.close();
    }
}

void Logger::send(const std::string& msg, int level) {
    if (shouldSend(level)) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::string level_str;
        switch(level) {
            case 0: level_str = "INFO"; break;
            case 1: level_str = "WARN"; break;
            case 2: level_str = "ERROR"; break;
            default: level_str = "L" + std::to_string(level); break;
        }
        
        file_ << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d:%H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << ms.count()
              << "] " << level_str << ": " << msg << std::endl;
    }
}

bool Logger::shouldSend(int level) const {
    return level >= min_level_;
}

} // namespace entropy