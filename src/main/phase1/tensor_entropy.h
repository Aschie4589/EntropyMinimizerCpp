#ifndef TENSOR_ENTROPY_H
#define TENSOR_ENTROPY_H

#include "cuda_traits.h"


template<typename T>
class TensorEntropyEstimator{
    /*
    
    TensorEntropyEstimator

    Responsible for computing the entropy of channels of the form Phi (x) \bar{Phi}, where Phi is has haar random unitaries as kraus operators.
    The MOE for Phi (x) \bar{Phi} can be estimated using Hastings argument: the maximally entangled vector is close to optimal with high probability.
    TensorEntropyEstimator computes the output entropy of the maximally entangled vector efficiently using properties of tensor products and partial traces.

    Strategy:
    TensorEntropyEstimator computes the complementary channel of Phi (x) \bar{Phi}:

    (Phi (x) \bar{Phi})^C [rho] = 1/(Nd^2)\sum_{ijkl} Tr[U_l U_k^* U_i U_J^*] |ij><kl|

    where U_i are the haar random unitaries used as kraus operators of Phi, and d is their number. (note: unitaries are not rescaled here!)

    TensorEntropyEstimator uses internally a CUDA kernel to compute the partial trace efficiently on GPU. If CUDA is not available, it falls back to a CPU implementation (very slow).
    It computes the entropy of the shifted channel: (Phi (x) \bar{Phi})^C_eps = (1-eps)(Phi (x) \bar{Phi})^C + eps Tr[rho] I/(d^2)

    It is also responsible for producing an estimate on the error made by shifting the channel.

    */
    public:
        TensorEntropyEstimator();
        ~TensorEntropyEstimator();

        // Methods to compute entropy of a tensor
        int computeEntropy(typename CudaTraits<T>::Complex* d_kraus, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, typename CudaTraits<T>::Real eps);

        // Getters for results
        typename CudaTraits<T>::Real getEpsilonEntropy();
        typename CudaTraits<T>::Real getEstimatedEntropy();
        typename CudaTraits<T>::Real getEstimatedError();


    private:
        typename CudaTraits<T>::Real epsilon; // Shift applied to the channel
        typename CudaTraits<T>::Real epsilon_entropy; // Entropy of the shifted channel
        typename CudaTraits<T>::Real estimated_entropy; // Estimated entropy of the original channel
        typename CudaTraits<T>::Real estimated_error; // Estimated error made by shifting the channel (that is entropy lies within [estimated_entropy - estimated_error, estimated_entropy + estimated_error])

        // Private helper methods and members can be defined here
};



#endif