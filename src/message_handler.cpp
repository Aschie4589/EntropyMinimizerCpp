#include "common_includes.h"
#include "message_handler.h"
#include "logger.h"
#include "printer.h"
MessageHandler::MessageHandler(/* args */)
{
    // Message handler simply broadcasts a message to all its loggers and or printers, depending on its settings.
    logging = true;
    printing = true;
}

int MessageHandler::createLogger(const std::string& f){
    loggers.emplace_back(new Logger(f));
    return 0;
}

int MessageHandler::createPrinter(){
    printers.emplace_back(new Printer());
    return 0;
}

int MessageHandler::setLogging(bool logging){
    this->logging = logging;
    return 0;
}

int MessageHandler::setPrinting(bool printing){
    this->printing = printing;
    return 0;
}

int MessageHandler::message(const std::string& message, int log_level){
    if (logging && !loggers.empty()){
        //Broadcast to loggers
        for (int i=0; i<loggers.size(); i++){
          loggers.at(i)->logMessage(message,log_level);
        }      
    }
    if (printing && !printers.empty()){
        for (int i=0; i<printers.size(); i++){
            printers.at(i)->printMessage(message, log_level);
        }
    }
    return 0;
}

MessageHandler::~MessageHandler()
{
    // Need to delete all printers and all loggers else I get memory leak.
    for (int i=0; i<loggers.size(); i++){
        delete loggers.at(i);
    }
    for (int i=0; i<printers.size(); i++){
        delete printers.at(i);
    }
}
