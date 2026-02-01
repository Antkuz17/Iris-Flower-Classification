#include <cmath>
#include "headers.hpp"


inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


class NeuralNetwork {
    public:

    private:
    

}



double mean_squared_error(const double* prediction, const double* actual, int size){

    // Take each output of the neural network and get the difference, tells you how bad the network is
    double error = std::pow(prediction[0] - actual[0], 2) +
               std::pow(prediction[1] - actual[1], 2) +
               std::pow(prediction[2] - actual[2], 2);
               
    return error/2.0;

}