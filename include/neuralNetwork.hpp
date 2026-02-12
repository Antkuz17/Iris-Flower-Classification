#ifndef neuralNetwork_HPP
#define neuralNetwork_HPP

#include <vector>
#include "matrix.hpp"
#include "dataExtract.hpp"


struct GradientStruct {
    Matrix dW1;
    Matrix dW2;
    Matrix db1;  
    Matrix db2;
};

class NeuralNetwork {
    public:

        // Constructor for the neural network
        NeuralNetwork(unsigned int inputNum, unsigned int outputNum, unsigned int HiddenLayerNum);

        // Forward propagation function that takes in an input vector and returns the output of the network as a vector of doubles
        Matrix forward_propagation(const Matrix& input);

        // Back propagation function that will return a struct containing the gradients of the weights and biases of the network based on the input, expected output, and actual output
        GradientStruct back_propagation(const Matrix& input, const Matrix& expected_output);

        // Train the neural network on a given dataset for a specified number of epochs and learning rate
        void train(const std::vector<std::vector<Record>>& training_data, int epochs, double learning_rate);

        // Test the neural network on a given dataset and print the accuracy of the network
        void test(const std::vector<std::vector<Record>>& testing_data);

        // Update the weights and biases of the network based on the calculated gradients and the learning rate
        void update_weights(const GradientStruct& gradients, double learning_rate);

    private:

        // The number of input, output, and hidden layer nodes
        unsigned int inputNum;
        unsigned int outputNum;
        unsigned int hiddenLayerNum;
        
        // The weights of the network, initialized in the constructor
        Matrix W1;
        Matrix W2;

        // The biases of the network, initialized in the constructor
        Matrix b1;
        Matrix b2;

};

inline double sigmoid(double x);
double mean_squared_error(const double* prediction, const double* actual, int size);

#endif