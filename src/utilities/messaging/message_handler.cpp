#include "utilities/messaging/message_handler.h"

void MessageHandler::addSink(std::unique_ptr<MessageSink> sink) {
        std::lock_guard<std::mutex> lock(mutex_);
        sinks_.push_back(std::move(sink));
}

void MessageHandler::send(const std::string& msg, int level) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& sink : sinks_) {
        sink->send(msg, level);
    }
}

void MessageHandler::clearSinks() {
    std::lock_guard<std::mutex> lock(mutex_);
    sinks_.clear();  // unique_ptr automatically deletes
}

size_t MessageHandler::getSinkCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return sinks_.size();
}