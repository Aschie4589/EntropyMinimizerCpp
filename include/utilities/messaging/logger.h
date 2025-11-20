#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <stdexcept>

#include "utilities/messaging/message_sink.h"
#include "utilities/uuid/uuid.h"

class Logger : public MessageSink {
private:
    std::ofstream file_;
    std::string uuid_;
    int min_level_;  // Only log messages >= this level
    
public:
    // Constructor
    Logger(const std::string& filename, int min_level = 0);    
    // Destructor
    ~Logger() override;
    
    // Implement the interface
    void send(const std::string& msg, int level) override; 
    
    bool shouldSend(int level) const override;
};


#endif