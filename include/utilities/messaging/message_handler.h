#ifndef MESSAGE_HANDLER_H
#define MESSAGE_HANDLER_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>

// Message Sink includes
#include "utilities/messaging/message_sink.h"

namespace utils {

/*
MessageHandler - Manages multiple message sinks (loggers, printers, etc.)
Features:
- Thread-safe addition/removal of sinks
- Broadcast messages to all registered sinks
- Convenience methods for different log levels
- Clear encapsulation of sink management: MessageHandler does not need to know about specific sink implementations or types
*/

class MessageHandler {
private:
    std::vector<std::unique_ptr<MessageSink>> sinks_; // Owned sinks (loggers, printers, etc.), using unique_ptr for automatic memory management
    std::mutex mutex_;  // Thread safety (lock access to sinks_)
    
public:
    MessageHandler() = default;
    
    // No copy (unique_ptr is not copyable)
    MessageHandler(const MessageHandler&) = delete;
    MessageHandler& operator=(const MessageHandler&) = delete;
    
    // Move semantics work automatically
    MessageHandler(MessageHandler&&) = default;
    MessageHandler& operator=(MessageHandler&&) = default;
    
    // Add a sink with ownership transfer
    void addSink(std::unique_ptr<MessageSink> sink);
    
    // Send message to ALL sinks (they filter themselves)
    void send(const std::string& msg, int level = 0);
    
    // Convenience method for different log levels
    // Level hierarchy: debug(-1) < info(0) < warn(1) < error(2)
    void debug(const std::string& msg) { send(msg, -1); }
    void info(const std::string& msg)  { send(msg, 0); }
    void warn(const std::string& msg)  { send(msg, 1); }
    void error(const std::string& msg) { send(msg, 2); }
    
    // Remove all sinks
    void clearSinks();
    
    size_t getSinkCount();
};

} // namespace utils

#endif