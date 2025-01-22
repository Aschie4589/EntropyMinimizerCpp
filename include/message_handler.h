#ifndef MESSAGE_HANDLER_H
#define MESSAGE_HANDLER_H

#include "common_includes.h"
#include "logger.h"
#include "printer.h"

/*
MessageHandler should do the following:
- if initialized as a logger, create a log file at a specified destination. then relay all messages to that log file.
- if intialized as a printer, simply print the messages passed.
*/
class MessageHandler
{
public:
    MessageHandler();
    ~MessageHandler();

    int createLogger(const std::string& filename);
    int createLogger();
    int createPrinter();

    int setLogging(bool logging);
    int setPrinting(bool printing);

    int message(const std::string& message, int log_level=LOG_LEVEL_INFO);
private:
    bool logging, printing;
    /* data */
    std::vector<Logger*> loggers;
    std::vector<Printer*> printers;
};



#endif