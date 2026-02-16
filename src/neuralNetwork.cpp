#include <cmath>
#include "neuralNetwork.hpp"
#include "matrix.hpp"

// Sigmoid activation function that takes in a double and returns the sigmoid of that double`
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Constructor for the neural network, initializes the weights and biases of the network
NeuralNetwork::NeuralNetwork(unsigned int input_size, unsigned int hidden_size, unsigned int output_size){

    // Initializing the meta data for the object like the all the sizes of the intermediary layers
    inputNum = input_size;
    outputNum = output_size;
    hiddenLayerNum = hidden_size;
        
    /// Initializing the weights and biases of the network, these are randomly generated between 0 and 0.1
    W1 = Matrix(input_size, hidden_size);
    W2 = Matrix(hidden_size, output_size);
    b1 = Matrix(1, hidden_size);
    b2 = Matrix(1, output_size);
}

NeuralNetwork::NeuralNetwork(){
    // Default constructor, initializes the network with 4 input nodes, 5 hidden layer nodes, and 3 output nodes
    inputNum = 4;
    outputNum = 3;
    hiddenLayerNum = 5;
        
    /// Initializing the weights and biases of the network, these are randomly generated between 0 and 0.1
    W1 = Matrix(inputNum, hiddenLayerNum);
    W2 = Matrix(hiddenLayerNum, outputNum);
    b1 = Matrix(1, hiddenLayerNum);
    b2 = Matrix(1, outputNum);
}


// The loss/cost function that compares the output to the expected
// In summary, this tells you how bad the network is performance wise
double mean_squared_error(const double* prediction, const double* actual, int size){
    double error = 0.0;
    for(int i = 0; i < size; i++){
        error += std::pow(prediction[i] - actual[i], 2);
    }
    return error / size;  // Or / 2.0, depending on your derivative convention
}

// Forward propagation function that takes in an input vector and returns the output of the network as a vector of doubles
Matrix NeuralNetwork::forward_propagation(const Matrix& input){
    // Save input for backprop
    input_cache = input;
    
    // Calculate hidden layer
    z1_cache = (input * W1) + b1;
    a1_cache = z1_cache.apply_function(sigmoid);

    // Calculate output layer
    z2_cache = (a1_cache * W2) + b2;
    Matrix a2 = z2_cache.apply_function(sigmoid);

    return a2;
}

// // Back propagation function that will return a struct containing the gradients of the weights and biases of the network based on the input, expected output, and actual output
// GradientStruct NeuralNetwork::back_propagation(const Matrix& input, const Matrix& expected_output){
//     // TODO: Logic
// }

// // Train the neural network on a given dataset for a specified number of epochs and learning rate
// void NeuralNetwork::train(const std::vector<std::vector<Record>>& training_data, int epochs, double learning_rate){
//     // TODO: Logic
// }

// // Test the neural network on a given dataset and print the accuracy of the network
// void NeuralNetwork::test(const std::vector<std::vector<Record>>& testing_data){
//     // TODO: Logic
// }

// // Update the weights and biases of the network based on the calculated gradients and the learning rate
// void NeuralNetwork::update_weights(const GradientStruct& gradients, double learning_rate){
//     // TODO: Logic
// }