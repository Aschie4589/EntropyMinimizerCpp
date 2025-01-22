#ifndef ENTROPY_CONFIG_H
#define ENTROPY_CONFIG_H

#include "common_includes.h"

class EntropyConfig {

    public:
        // Algorithm config
        int max_iterations, minimization_attempts;
        double epsilon;
        // Specific to prediction of final entropy of a run
        bool MOE_use_prediction;
        double MOE_prediction_tolerance;


        // Messages config

        bool log, print;
        std::string log_file;
        bool use_custom_log_file;


        EntropyConfig();

        // Setters
        int setEpsilon(double eps);
        int setLogging(bool l);
        int setPrinting(bool p);
        int setLogFile(const std::string& lf);
        int setMOEUsePrediction(bool mup);
        int setMOEPredictionTolerance(double mpt);

};

#endif