#include "common_includes.h"
#include "entropy_estimator.h"
#include "config.h"
#include "matrix_operations.h"
EntropyEstimator::EntropyEstimator(){
    // Define the parameters
    window_size = ENTROPY_ESTIMATOR_DEFAULT_WINDOW_SIZE;
    current_window_index = 0;

    window = new double[max_window_size];
    deltas = new double[max_window_size-1];

    model_params = new double[2]; // slope and intercept of line
    X_matrix = new double[2*max_window_size-2];
    Y_vector = new double[max_window_size-1];
}



int EntropyEstimator::appendEntropy(double data){
    // Append the entropy data point to the array
    window[current_window_index % max_window_size] = data;
    // Also append the log of delta entropy to the other array
    if (current_window_index > 0){
       deltas[(current_window_index-1) % (max_window_size-1)] = std::log(window[(current_window_index-1) % max_window_size]-data);
    }
    // Increase the current index by 1
    current_window_index ++;

    return 0;
}

double* EntropyEstimator::getLastEntropy(){
    return &window[(current_window_index-1) % max_window_size];
}


// HERE
int EntropyEstimator::updateXY(){
    // Update the X matrix. First entries are just 1.
    for (int i=0; i < window_size-1; i++){
       X_matrix[i] = 1.0f; 
    }
    // The remaining entries record the iteration. Iteration is 1 less than the current_window_index, so that iteration=1 corresponds to 1 entry in the deltas buffer.
    // There will be overflows because of size_t being unsigned! So X contains garbage until at least window_size-1 data points have been added!
    for (int i=0; i < window_size-1; i++){
        X_matrix[window_size-1+i] = (current_window_index-window_size+i+1);
    }
    // Update the Y matrix. Copy the data over from the delta buffer.
    // Notice that for the first few entries, there will be an overflow! size_t is unsigned, so going negative means looping back. This gives a very high number which is not necessarily congruent to the right thing!
    for (int i=0; i< window_size-1; i++){
        Y_vector[i] = deltas[(current_window_index-window_size+i) % (max_window_size-1)];
    }

    return 0;
}

/*
This method performs exponential fit. I.e. it performs linear fit with the deltas matrix, which contains the log of the deltas. 
Returns R^2 value, -1 if unsuccessful.
If successful, model_params now contains the updated linear fit parameters.
*/
double EntropyEstimator::exponentialFit(){
    // Step 1: check if enough data points have been obtained, so as not to give whack results
    if (current_window_index < window_size){
        //std::cout << "Not enough data has been collected to perform exponential fit..." << std::endl;
        return 0;
    }
    // Step 2: update the datas_matrix
    updateXY();
    // Step 3: calculate (X_matrix^T * X_matrix) and X_matrix^T* Y_vector. Then calculate the model parameters by solving the linear system XTX * model_params = XTY
    // First, calculate X_matrix^T * X_matrix using BLAS
    double* XTX = new double[4];
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 2, 2, window_size-1, 1.0, X_matrix, window_size-1, X_matrix, window_size-1, 0.0, XTX, 2);
    // Next, calculate X_matrix^T * Y_vector
    double* XTY = new double[2];
    cblas_dgemv(CblasColMajor, CblasTrans, window_size-1, 2, 1.0, X_matrix, window_size-1, Y_vector, 1, 0.0, XTY, 1);
    // Finally, solve the linear system using LAPACK and the wrapper
    int ipiv[2];

    dgesv_wrapper(2, 1, XTX, 2, ipiv, XTY, 2);
    // Copy the model parameters over
    model_params[0] = XTY[0];
    model_params[1] = XTY[1];
        
    // Clean up
    delete[] XTX;
    delete[] XTY;

    // Step 4: calculate R^2 value
    // First, calculate the mean of the Y_vector
    double mean_Y = 0.0;
    for (int i=0; i<window_size-1; i++){
        mean_Y += Y_vector[i];
    }
    mean_Y /= window_size-1;
    // Next, calculate the total sum of squares
    double TSS = 0.0;
    for (int i=0; i<window_size-1; i++){
        TSS += (Y_vector[i] - mean_Y)*(Y_vector[i] - mean_Y);
    }
    // Next, calculate the residual sum of squares
    double RSS = 0.0;
    for (int i=0; i<window_size-1; i++){
        double y_pred = model_params[0] + model_params[1]*X_matrix[window_size-1+i];
        RSS += (Y_vector[i] - y_pred)*(Y_vector[i] - y_pred);
    }
    // Finally, calculate R^2
    double R2 = 1.0 - RSS/TSS;


    return R2;
}

/*
Without updating the model parameters, predict the final entropy value. 
If the model is y=ax+b, then the final improvement is exp(window[-1])*a/(1-a)
The predicted entropy is then window[-1] - improvement
*/
double EntropyEstimator::predictFinalEntropy(){
    // Step 1: check if enough data points have been obtained, so as not to give whack results
    if (current_window_index < window_size){
        //std::cout << "Not enough data has been collected to predict final entropy..." << std::endl;
        return -1.0;
    }
    // Also if the slope is positive, we are in trouble. The model is not valid.
    if (model_params[1] > 0){
        //std::cout << "The model is not valid for prediction. The slope is positive..." << std::endl;
        return -1.0;
    }
    // Step 2: calculate the improvement
    double improvement = std::exp(deltas[(current_window_index-1) % max_window_size])*std::exp(model_params[1])/(1.0-std::exp(model_params[1]));
    // Step 3: return the predicted entropy
    return window[(current_window_index-1) % max_window_size] - improvement;
}

int EntropyEstimator::predictFinalSteps(){
    // Step 1: check if enough data points have been obtained, so as not to give whack results
    if (current_window_index < window_size){
        //std::cout << "Not enough data has been collected to predict final steps..." << std::endl;
        return -1.0;
    }
    // Also if the slope is positive, we are in trouble. The model is not valid.
    if (model_params[1] > 0){
        //std::cout << "The model is not valid for prediction. The slope is positive..." << std::endl;
        return -1.0;
    }
    // Step 2: calculate the predicted steps
    int predicted_steps = std::round((std::log(CONVERGENCE_TOLERANCE)-model_params[0])/model_params[1]);
    // Step 3: return the predicted steps
    return predicted_steps;
}

int EntropyEstimator::reset(){
    // Reset the entropy estimator to its initial state
    current_window_index = 0;
    return 0;
}

EntropyEstimator::~EntropyEstimator(){
    delete[] window;
    delete[] deltas;
    delete[] model_params;
    delete[] X_matrix;
    delete[] Y_vector;
}
