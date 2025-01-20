#ifndef ENTROPY_ESTIMATOR_H
#define ENTROPY_ESTIMATOR_H

#include "common_includes.h"
#include "config.h"

class EntropyEstimator{
    public:
        EntropyEstimator();

        ~EntropyEstimator();

        int appendEntropy(double data);
        double* getLastEntropy();
        double exponentialFit(); // This method performs exponential fit, then returns the R^2 value of the model. model_params are updated.
        double predictFinalEntropy(); // This method predicts the final entropy value based on the model. The model is y=ax+b, so the final improvement is exp(window[-1])*a/(1-a). The predicted entropy is then window[-1] + improvement
        int predictFinalSteps(); // This method predicts the number of steps to reach the final entropy value. The model is y=ax+b, and the predicted steps are (log(tolerance)-b)/a

        int updateXY();
        int reset(); // This resets the entropy estimator to its initial state

        size_t window_size, current_window_index;
        double* deltas; // This stores the delta entropies over time. It is one shorter than window size.


        static constexpr size_t max_window_size = ENTROPY_ESTIMATOR_MAX_WINDOW_SIZE; // This defines a constant expression that is shared among all instances of the class
        double* window; // This is a pointer to an array of complex numbers. The array parentheses are not necessary, since the pointer only needs to point to the first element of the array.

        // HERE   
        double* X_matrix; // This is the X matrix of linear regression. Of the form [1 | N], where 1 is a column of 1s and N contains the iteration number
        double* Y_vector; // This stores the last available deltas
        double* model_params; // These are the parameters (intercept, slope) to be updated over time
    private:

};


#endif