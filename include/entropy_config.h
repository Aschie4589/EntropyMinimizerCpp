#ifndef ENTROPY_CONFIG_H
#define ENTROPY_CONFIG_H

class EntropyConfig {

    public:
        // Algorithm config
        int max_iterations;
        double epsilon;

        // Messages config

        bool log, print;
        std::string log_file;

        EntropyConfig();

        // Setters
        int setEpsilon(double eps);
        int setLogging(bool l);
        int setPrinting(bool p);
        int setLogFile(const std::string& lf);

};

#endif