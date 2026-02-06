#include <cmath>
#include "headers.hpp"


inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


class NeuralNetwork {
    public:
        NeuralNetwork(unsigned int inputNum, unsigned int outputNum, unsigned int HiddenLayerNum);

        Matrix transpose() const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const
        Matrix elementwise_multiply(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        Matrix apply_function(double (*func)(double)) const;

    private:
    

}


double mean_squared_error(const double* prediction, const double* actual, int size){

    // Take each output of the neural network and get the difference, tells you how bad the network is
    double error = std::pow(prediction[0] - actual[0], 2) +
               std::pow(prediction[1] - actual[1], 2) +
               std::pow(prediction[2] - actual[2], 2);
               
    return error/2.0;

}