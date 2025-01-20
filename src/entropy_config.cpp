#include "common_includes.h"

#include "entropy_config.h"
#include "config.h"

EntropyConfig::EntropyConfig() {
    // Set default values

    //Algorithm
    max_iterations = DEFAULT_MINIMIZER_MAX_ITERATIONS;
    epsilon = DEFAULT_MINIMIZER_EPSILON;
    minimization_attempts = DEFAULT_MINIMIZER_MINIMIZATION_ATTEMPTS;

    // MOE prediction
    MOE_use_prediction = DEFAULT_MINIMIZER_USE_MOE_PREDICTION;
    MOE_prediction_tolerance = DEFAULT_MINIMIZER_MOE_PREDICTION_TOLERANCE;

    //Logging and messaging
    log = DEFAULT_MINIMIZER_LOG;
    log_file = DEFAULT_MINIMIZER_LOG_FILENAME;
    print = DEFAULT_MINIMIZER_PRINT;
    


}

int EntropyConfig::setMOEUsePrediction(bool mup){
    MOE_use_prediction = mup;
    return 0;
}

int EntropyConfig::setMOEPredictionTolerance(double mpt){
    MOE_prediction_tolerance = mpt;
    return 0;
}

int EntropyConfig::setEpsilon(double eps){
    epsilon = eps;
    return 0;
}

int EntropyConfig::setLogging(bool l){
    log = l;
    return 0;
}

int EntropyConfig::setPrinting(bool p){
    print = p;
    return 0;
}

int EntropyConfig::setLogFile(const std::string& lf){
    log_file = lf;
    return 0;
}