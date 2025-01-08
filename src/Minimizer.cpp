// Minimizer.cpp
#include "common_includes.h"
#include "minimizer.h"
#include "config.h"


Minimizer::Minimizer(std::vector<std::complex<double> >* kraus_ops, 
                        int kraus_number, int kraus_in_dimension, int kraus_out_dimension,double eps) {
    // PARAMETERS ASSIGNMENT
    // kraus_operators stores the pointer to a vector of double precision complex numbers. I don't want to have two large objects in memory so this is the way to go.
    // It consists of a list of d contiguous NxN matrices, each stored in column-major order.
    // So entry (i,j) of matrix k corresponds to index k*N*N+j*N+i
    // TODO: introduce a check that kraus_ops has the right size?
    kraus_operators = kraus_ops;
    // assign the dimensions. Since I assign values, those are copied and d, N, M live in memory where the class instance lives
    d = kraus_number;
    N = kraus_in_dimension;
    M = kraus_out_dimension;
    // assign precision
    epsilon = eps; 

    // INITIALIZATION OF MATRICES AND VECTORS
    // start by initializing the vector_state to a zero vector, to avoid seg faults.
    vector_state = new std::vector(N,std::complex<double>(0.0f,0.0f));
    // also initialize the input and output matrices to zero vectors, again for safety.
    input_matrix = new std::vector(N*N,std::complex<double>(0.0f,0.0f));
    output_matrix = new std::vector(M*M,std::complex<double>(0.0f,0.0f));

    // OTHER USEFUL CONSTANTS
    entropy = -1;
    // we need to know the binary entropy of epsilon, which is used in obtaining the error in the approximation of the entropy
    bin_entropy = -epsilon * std::log(epsilon) - (1-epsilon) * std::log(1-epsilon);
    // the corresponding error is quantified by bin_ent(eps)/2*(1-eps)
    entropy_error = bin_entropy / (2*(1-epsilon));
}

int Minimizer::initializeVector(std::vector<std::complex<double> >* pointer) {
    // By construction, vector_state is always a vector that we have created with "new".
    // We want to keep that pointer and just copy the data over, if "pointer" is a valid
    // reference to a new vector of the right size

    // Step 1: Check that the pointer points to a valid, non-empty vector:
    if (pointer == nullptr || pointer->empty()){
        std::cout << "Wait a second, no starting vector was provided! I will generate a random one..." << std::endl;
        return initializeRandomVector();
    } 
    // Step 2: Check that the vector has the right dimension (equal to the input dimension)
    if (pointer->size()==N){
        std::cout << "The input vector has the right dimension" << std::endl;
        // Copy data over, keep the pointer
        *vector_state = *pointer;
        return 0;
    } else {
        // If it doesn't, generate a random vector as a fallback.
        std::cout << "The vector does not match the specified dimension! Fallback: generating random vector..." << std::endl;
        return initializeRandomVector();
    }
}

int Minimizer::initializeRandomVector(){
    // By construction, the pointer points to a valid vector of the right size. Note that N is fixed and cannot change.

    // Step 1: Set up random number generator
    std::random_device rd;                  // Random device to seed the generator
    std::mt19937 gen(rd());                 // Mersenne Twister generator, seeded by rd()
    std::normal_distribution<double> dist(0.0f, 1.0f); // Normal distribution with mean and stddev

    // Step 2: Generate random real and imaginary parts
    for (int i=0; i < N; i++){
        double real_part = dist(gen);  // Generate the real part (normal distribution)
        double imag_part = dist(gen);  // Generate the imaginary part (normal distribution)
        vector_state->at(i) = std::complex<double>(real_part, imag_part);
    }

    // Step 3: Renormalize the vector
    double norm = 0.0f;
    for (std::complex<double> entry : *vector_state){
        norm += std::pow(std::abs(entry),2);
    }
    norm = std::sqrt(norm);
    // std::cout << "Norm of the vector is: "<<norm<<std::endl;
    if (norm != 0) {
        for (int i = 0; i < vector_state->size(); i++){
            vector_state->at(i) /= norm;
        }
    }
    return 0;
}

int Minimizer::updateProjector(){
    // We use column-major order, to comply with LAPACK
    for (int i=0;i<N;i++){ // Go through rows
        for (int j=0; j<N; j++){
            // Entry (i,j) of matrix (row i, col j) is stored at j*N+i (0-based)
            // Entry (i,j) is |i><j|
            input_matrix->at(i+j*N) = vector_state->at(i) * std::conj(vector_state->at(j));
        }
    }
    return 0;
}

int Minimizer::applyChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension){
    // The three pointers are for: - where kraus ops are stored; - where input matrix is stored; where output matrix needs to be stored
    // Then we need to know how many kraus operators there are, and what is their size.
    lapack_complex_double one(1.0f+0.0fi);
    lapack_complex_double zero(0.0f+0.0fi);

    std::vector<std::complex<double> > tmp_matrix = std::vector(in_dimension*out_dimension,std::complex<double>(0.0f,0.0f));
    std::vector<std::complex<double> >* tmp_pointer = &tmp_matrix;

    // Step 0: make sure that out_matrix contains only zeros.
    std::fill(out_matrix->begin(), out_matrix->end(), std::complex<double>(0.0f,0.0f));

    // Iterate over all kraus operators
    for (int m=0; m<number_kraus; m++){
        // Work with the m-th Kraus operator (update the pointer)
        std::complex<double>* kraus_pointer = &(kraus->at(m*in_dimension*out_dimension));
  
        // Step 1: create temporary value Kraus[i] * in_matrix. Save in "channel_application_intermediate matrix".
        cblas_zgemm(CblasColMajor,
            CblasNoTrans,                      // No transpose for A
            CblasNoTrans,                      // No transpose for B
            out_dimension, in_dimension, in_dimension,                  // Matrix dimensions (#rows of C, #cols of C, contracted dimension)
            &one,                    // Scalar alpha
            reinterpret_cast<lapack_complex_double*>(kraus_pointer),  // Matrix A
            out_dimension,                        // Leading dimension of A (=how many elements to skip to get to next column) (=number of rows). K is MxN so it has M rows
            reinterpret_cast<lapack_complex_double*>(in_matrix->data()),  // Matrix B
            in_dimension,                        // Leading dimension of B. in_matrix is NxN so N
            &zero,                     // Scalar beta
            reinterpret_cast<lapack_complex_double*>(tmp_pointer->data()),  // Matrix C (result)
            out_dimension                         // Leading dimension of C. C is MxN so M
        );
        // Step 2: add channel_application_intermediate_matrix * conj(Kraus^T)=K[i]*in*conj(K^T) to out_matrix.
        cblas_zgemm(CblasColMajor,
            CblasNoTrans,                   // No transpose for A
            CblasConjTrans,                      // Transpose conjugate for B (need Kraus^H)
            out_dimension, in_dimension, in_dimension,                  // Matrix dimensions (#rows of C, #cols of C, contracted dimension)
            &one,                    // Scalar alpha
            reinterpret_cast<lapack_complex_double*>(tmp_pointer->data()),  // Matrix A
            out_dimension,                        // Leading dimension of A (=how many elements to skip to get to next column) (=number of rows). K*in_matrix is MxN so it has M rows
            reinterpret_cast<lapack_complex_double*>(kraus_pointer),  // Matrix B
            in_dimension,                        // Leading dimension of B. K^H is NxM, so N
            &one,                     // Scalar beta
            reinterpret_cast<lapack_complex_double*>(out_matrix->data()),  // Matrix C (result)
            out_dimension                         // Leading dimension of C. C is MxM so M
        );
    }
    return 0;
}
    
int Minimizer::applyDualChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension){
    // The three pointers are for: - where kraus ops are stored; - where input matrix is stored; where output matrix needs to be stored
    // Then we need to know how many kraus operators there are, and what is their size.
    lapack_complex_double one(1.0f+0.0fi);
    lapack_complex_double zero(0.0f+0.0fi);

    std::vector<std::complex<double> > tmp_matrix = std::vector(in_dimension*out_dimension,std::complex<double>(0.0f,0.0f));
    std::vector<std::complex<double> >* tmp_pointer = &tmp_matrix;

    // Step 0: make sure that out_matrix contains only zeros.
    std::fill(out_matrix->begin(), out_matrix->end(), std::complex<double>(0.0f,0.0f));

    // Iterate over all kraus operators
    for (int m=0; m<number_kraus; m++){
        // Work with the m-th Kraus operator (update the pointer)
        std::complex<double>* kraus_pointer = &(kraus->at(m*in_dimension*out_dimension));
  
        // Step 1: create temporary value Kraus[i] * in_matrix. Save in "channel_application_intermediate matrix".
        cblas_zgemm(CblasColMajor,
            CblasConjTrans,                      // No transpose for A
            CblasNoTrans,                      // No transpose for B
            out_dimension, in_dimension, in_dimension,                  // Matrix dimensions (#rows of C, #cols of C, contracted dimension)
            &one,                    // Scalar alpha
            reinterpret_cast<lapack_complex_double*>(kraus_pointer),  // Matrix A
            out_dimension,                        // Leading dimension of A (=how many elements to skip to get to next column) (=number of rows). K is MxN so it has M rows
            reinterpret_cast<lapack_complex_double*>(in_matrix->data()),  // Matrix B
            in_dimension,                        // Leading dimension of B. in_matrix is NxN so N
            &zero,                     // Scalar beta
            reinterpret_cast<lapack_complex_double*>(tmp_pointer->data()),  // Matrix C (result)
            out_dimension                         // Leading dimension of C. C is MxN so M
        );
        // Step 2: add channel_application_intermediate_matrix * conj(Kraus^T)=K[i]*in*conj(K^T) to out_matrix.
        cblas_zgemm(CblasColMajor,
            CblasNoTrans,                   // No transpose for A
            CblasNoTrans,                      // Transpose conjugate for B (need Kraus^H)
            out_dimension, in_dimension, in_dimension,                  // Matrix dimensions (#rows of C, #cols of C, contracted dimension)
            &one,                    // Scalar alpha
            reinterpret_cast<lapack_complex_double*>(tmp_pointer->data()),  // Matrix A
            out_dimension,                        // Leading dimension of A (=how many elements to skip to get to next column) (=number of rows). K*in_matrix is MxN so it has M rows
            reinterpret_cast<lapack_complex_double*>(kraus_pointer),  // Matrix B
            in_dimension,                        // Leading dimension of B. K^H is NxM, so N
            &one,                     // Scalar beta
            reinterpret_cast<lapack_complex_double*>(out_matrix->data()),  // Matrix C (result)
            out_dimension                         // Leading dimension of C. C is MxM so M
        );
    }
    return 0;
}

int Minimizer::applyEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon){
    // Step 1: apply the channel without the epsilon
    applyChannel(kraus,in_matrix,out_matrix,number_kraus,in_dimension,out_dimension);

    // Step 2: interpolate: scale the output of the previous operation by (1-epsilon), then add epsilon/d on the diagonal
    for (int i=0; i<out_matrix->size();i++){
        out_matrix->at(i) =  out_matrix->at(i)*std::complex<double>(1-epsilon,0.0f);
    }
    for (int i=0; i<out_dimension; i++){
        out_matrix->at(i*out_dimension+i) += std::complex<double>(epsilon, 0.0f)/std::complex<double>(out_dimension, 0.0f);
    }
    return 0;
}

int Minimizer::applyDualEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon){
    // Step 1: apply the channel without the epsilon
    applyDualChannel(kraus,in_matrix,out_matrix,number_kraus,in_dimension,out_dimension);

    // Step 2: interpolate: scale the output of the previous operation by (1-epsilon), then add epsilon on the diagonal
    for (int i=0; i<out_matrix->size();i++){
        out_matrix->at(i) *= (1-epsilon);
    }
    for (int i=0; i<out_dimension; i++){
        out_matrix->at(i*out_dimension+i) += std::complex<double>(epsilon, 0.0f)/std::complex<double>(out_dimension, 0.0f);
    }
    return 0;
}

int Minimizer::stepAlgorithm(){
    // Step 1: update the projector based on the vector
    updateProjector();

    // Step 2: compute Phi_e(rho)
    applyEpsilonChannel(kraus_operators,input_matrix, output_matrix, d, N, M, epsilon);

    // Step 3: compute log(Phi_e(rho))
    // Step 3.1: diagonalize output matrix
    std::vector<double> eigvals(M);
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', M, reinterpret_cast<lapack_complex_double*>(output_matrix->data()),M,eigvals.data());
    if (info != 0){
        std::cout << "Something went wrong in diagonalizing the matrix...!" <<std::endl;
        return 1;
    }
    // Step 3.2: compute log(eigs)
    std::vector<std::complex<double> >* eig_mat = new std::vector<std::complex<double> >(N*N,std::complex<double>(0.0f,0.0f));
    for (int i=0; i<M; i++){
        eig_mat->at(N*i+i) = std::log(eigvals.at(i));
    }

    // Step 3.3: reconstruct the matrix
    //output matrix contains the eigenvectors as columns. Perform matrix multiplication.
    // First perform diag * eigs, store result in another temp mat
    std::vector<std::complex<double> >* tmp_mat = new std::vector<std::complex<double> >(N*N);
    lapack_complex_double one(1.0f+0.0fi);
    lapack_complex_double zero(0.0f+0.0fi);
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,M,M,M,&one,reinterpret_cast<lapack_complex_double*>(eig_mat->data()),M,reinterpret_cast<lapack_complex_double*>(output_matrix->data()),M,&zero,reinterpret_cast<lapack_complex_double*>(tmp_mat->data()),M);
    // Then do conj(eigs^T)*tmp_mat
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,M,M,&one,reinterpret_cast<lapack_complex_double*>(output_matrix->data()),M,reinterpret_cast<lapack_complex_double*>(tmp_mat->data()),M,&zero,reinterpret_cast<lapack_complex_double*>(eig_mat->data()),M);
    for (int i=0; i< M*M; i++){
        output_matrix->at(i) = eig_mat->at(i);
    }
    delete eig_mat;
    delete tmp_mat;

    // As far as I can tell, the matrix logarithm is calculated correctly.

    // Step 4: compute Phi_e^*(log(Phi_e(rho)))
    applyDualChannel(kraus_operators,output_matrix,input_matrix,d,M,N);

    // Step 5: find eigenvector with highest eigenvalues
    eigvals.resize(N);
    info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', N, reinterpret_cast<lapack_complex_double*>(input_matrix->data()),N,eigvals.data());
    if (info != 0){
        std::cout << "Something went wrong in diagonalizing the matrix...!" <<std::endl;
        return 1;
    }

    

    // Update the vector state to the last column, which corresponds to the highest eigenvalue.
    for (int i=0; i< N; i++){
        vector_state->at(i) = input_matrix->at(N*(N-1)+i);
    }



    return 0;
}

int Minimizer::calculateEntropy(){
    // Get the vN entropy of Phi_e(state).
    // This works, pending verification on the application of the EpsilonChannel.

    updateProjector();
    applyEpsilonChannel(kraus_operators, input_matrix, output_matrix, d, N, M, epsilon);
    std::vector<double> eigvals = std::vector<double>(N);
    std::vector<std::complex<double> > tmp(M*M);
    for (int i=0; i < M*M; i++){
        tmp.at(i) = output_matrix->at(i);
    }
    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', M, reinterpret_cast<lapack_complex_double*>(tmp.data()),M,eigvals.data());
    entropy = 0.0f;
    for (int i = 0; i< M; i++){
        // WARNING: We are assuming that the diagonal here is real (which it is since it contains the eigs of a hermitian matrix)
        entropy -= eigvals.at(i)*std::log(eigvals.at(i));
    }
    //std::cout<< "Current entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) <<entropy << std::endl;

    return 0;

}

int Minimizer::step(){
    stepAlgorithm();
    calculateEntropy();
    return 0;
}

/// GETTERS
std::vector<std::complex<double> >* Minimizer::getState(){
    return input_matrix;
}

double* Minimizer::getEntropy(){
    return &entropy;
}

/// DEBUGGING AND PRINTING AND ETC

int Minimizer::printVectorState(){
    // Print a list. Some variables are hard coded!
    if (vector_state == nullptr){
        std::cout << "No assigned value for vector..." << std::endl;
        return 1;
    } else {
        int size = vector_state->size();
        std::cout << "[";
        for (int i = 0; i < size; i++) {
                double r = (*vector_state)[i].real();
                double j = (*vector_state)[i].imag();
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION) << r;
                if (j>-1e-10){
                    std::cout << "+";
                }
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION)<<j << "j, ";
        }
        std::cout << "]"<< std::endl;
        return 0;
    }

}

int Minimizer::printMatrix(std::vector<std::complex<double> >* matrix_pointer, int n, int m) {
    // "Pretty" prints a nxm matrix in column-major order
    // it has to loop over every row, then print every entry of the row by looping over columns
    // there are n rows and m columns
    // To begin with, let's just print the entries and separate them by tabs or something
    // All variables are hardcoded here!
    for (int i=0; i<n; i++){
        for (int j=0;j<m; j++){
                double re = (*matrix_pointer)[j*m+i].real();
                double im = (*matrix_pointer)[j*m+i].imag();
                if (re>-1e-15){
                    std::cout << " ";
                }
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION)<< re;
                if (im>-1e-15){
                    std::cout << "+";
                }
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION)<<im << "j ";
        }
        std::cout << std::endl;
    }
    return 0;
}

int Minimizer::printState(){
    printMatrix(getState(), N, N);
    return 0;
}

Minimizer::~Minimizer() {
    delete vector_state;
    delete input_matrix;
    delete output_matrix;
}