#include <cmath>
#include "neuralNetwork.hpp"
#include "matrix.hpp"
#include <iostream>

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
double mean_squared_error(const Matrix& prediction, const Matrix& actual){
    double error = 0.0;
    for(unsigned int j = 0; j < prediction.get_num_col(); j++){
        error += std::pow(prediction.get_val(0, j) - actual.get_val(0, j), 2);
    }
    return error / 2.0;
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

GradientStruct NeuralNetwork::back_propagation(const Matrix& input, const Matrix& expected_output){
    // Forward propagation to get intermediate values
    Matrix A2 = forward_propagation(input);
    
    // Output layer error
    Matrix dZ2 = A2 - expected_output;
    
    // Gradients for W2 and b2
    Matrix dW2 = a1_cache.transpose() * dZ2;
    Matrix db2 = dZ2;

    // sigmoid derivative = A1 * (1 - A1)
    Matrix ones(a1_cache.get_num_rows(), a1_cache.get_num_col());
    // Fill ones matrix with 1.0
    for(unsigned int i = 0; i < ones.get_num_rows(); i++){
        for(unsigned int j = 0; j < ones.get_num_col(); j++){
            ones.set_val(i, j, 1.0);
        }
    }
    Matrix sigmoid_deriv = a1_cache.elementwise_multiply(ones - a1_cache);
    Matrix dZ1 = (dZ2 * W2.transpose()).elementwise_multiply(sigmoid_deriv);
    
    Matrix dW1 = input.transpose() * dZ1;
    Matrix db1 = dZ1;
    
    return GradientStruct{dW1, db1, dW2, db2};
}

void NeuralNetwork::train(const std::vector<std::vector<Record>>& training_data, int epochs, double learning_rate){
    std::vector<Record> records = training_data[0];

    for(int epoch = 0; epoch < epochs; epoch++){
        double total_cost = 0.0;

        for(int i = 0; i < records.size(); i++){
            // Build input matrix for one sample
            Matrix X(1, inputNum, 0.0);
            X.set_val(0, 0, records[i].sepal_length);
            X.set_val(0, 1, records[i].sepal_width);
            X.set_val(0, 2, records[i].pedal_length);
            X.set_val(0, 3, records[i].pedal_width);

            // Build label matrix for one sample
            Matrix Y(1, outputNum, 0.0);
            Y.set_val(0, 0, records[i].one_hot[0]);
            Y.set_val(0, 1, records[i].one_hot[1]);
            Y.set_val(0, 2, records[i].one_hot[2]);

            // Forward pass
            Matrix A2 = forward_propagation(X);

            // Cost
            total_cost += mean_squared_error(A2, Y);

            // Backprop and update
            GradientStruct gradients = back_propagation(X, Y);
            update_weights(gradients, learning_rate);
        }

        if(epoch % 100 == 0){
            std::cout << "Epoch " << epoch << " cost: " << total_cost / records.size() << std::endl;
        }
    }
}


void NeuralNetwork::test(const std::vector<std::vector<Record>>& testing_data){
    std::vector<Record> records = testing_data[1];
    int correct = 0;

    for(int i = 0; i < records.size(); i++){
        // Build input matrix for one sample
        Matrix X(1, inputNum, 0.0);
        X.set_val(0, 0, records[i].sepal_length);
        X.set_val(0, 1, records[i].sepal_width);
        X.set_val(0, 2, records[i].pedal_length);
        X.set_val(0, 3, records[i].pedal_width);

        // Build label matrix for one sample
        Matrix Y(1, outputNum, 0.0);
        Y.set_val(0, 0, records[i].one_hot[0]);
        Y.set_val(0, 1, records[i].one_hot[1]);
        Y.set_val(0, 2, records[i].one_hot[2]);

        Matrix A2 = forward_propagation(X);

        // Find predicted class
        int predicted = 0;
        double max_val = A2.get_val(0, 0);
        for(int j = 1; j < outputNum; j++){
            if(A2.get_val(0, j) > max_val){
                max_val = A2.get_val(0, j);
                predicted = j;
            }
        }

        // Find actual class
        int actual = 0;
        for(int j = 1; j < outputNum; j++){
            if(Y.get_val(0, j) > Y.get_val(0, actual)){
                actual = j;
            }
        }

        if(predicted == actual) correct++;
    }

    double accuracy = (double)correct / records.size() * 100.0;
    std::cout << "Test accuracy: " << accuracy << "%" << std::endl;
}

// Update the weights and biases of the network based on the calculated gradients and the learning rate
void NeuralNetwork::update_weights(const GradientStruct& gradients, double learning_rate){
    W1 = W1 - gradients.dW1 * learning_rate;
    b1 = b1 - gradients.db1 * learning_rate;
    W2 = W2 - gradients.dW2 * learning_rate;
    b2 = b2 - gradients.db2 * learning_rate;
}