#include <cmath>
#include "neuralNetwork.hpp"

// Sigmoid activation function that takes in a double and returns the sigmoid of that double`
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Constructor for the neural network, initializes the weights and biases of the network
NeuralNetwork::NeuralNetwork(unsigned int inputNum, unsigned int outputNum, unsigned int HiddenLayerNum);


// The loss/cost function that compares the output to the expected
// In summary, this tells you how bad the network is performance wise
double NeuralNetwork::mean_squared_error(const double* prediction, const double* actual, int size){
    double error = 0.0;
    for(int i = 0; i < size; i++){
        error += std::pow(prediction[i] - actual[i], 2);
    }
    return error / size;  // Or / 2.0, depending on your derivative convention
}

// Forward propagation function that takes in an input vector and returns the output of the network as a vector of doubles
Matrix NeuralNetwork::forward_propagation(const Matrix& input);

// Back propagation function that will return a struct containing the gradients of the weights and biases of the network based on the input, expected output, and actual output
GradientStruct NeuralNetwork::back_propagation(const Matrix& input, const Matrix& expected_output);

// Train the neural network on a given dataset for a specified number of epochs and learning rate
void NeuralNetwork::train(const std::vector<std::vector<Record>>& training_data, int epochs, double learning_rate);

// Test the neural network on a given dataset and print the accuracy of the network
void NeuralNetwork::test(const std::vector<std::vector<Record>>& testing_data);

// Update the weights and biases of the network based on the calculated gradients and the learning rate
void NeuralNetwork::update_weights(const GradientStruct& gradients, double learning_rate);