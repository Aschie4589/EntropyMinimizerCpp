#ifndef PRINTER_H
#define PRINTER_H

#include <string>
#include <filesystem>

#include "utilities/messaging/message_sink.h"
#include "utilities/uuid/uuid.h"

class Printer : public MessageSink {
private:
    std::string uuid_;
    int min_level_;
    bool color_enabled_;
    
public:
    Printer(int min_level = 0, bool color = true);
    
    void send(const std::string& msg, int level) override;

    bool shouldSend(int level) const override; 
    
private:
    std::string getLevelColor(int level) const;

};

#endif