#ifndef MESSAGE_SINK_H
#define MESSAGE_SINK_H

#include <string>
/*
MessageSink - Abstract base class for message sinks (loggers, printers, etc.)
*/
class MessageSink {
public:
    virtual ~MessageSink() = default;  // Virtual destructor for proper cleanup
    
    // Pure virtual method - all sinks must implement
    virtual void send(const std::string& msg, int level) = 0;
    
    // Optional: filtering capability
    virtual bool shouldSend(int level) const { return true; }
};

#endif // MESSAGE_SINK_H