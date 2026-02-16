// Gonna recode the entire network in c++ 
// Why? cause i love to suffer and feel pain
#include "visualizer.hpp"
#include "matrix.hpp"
#include "neuralNetwork.hpp"
#include "dataExtract.hpp"

#include <vector>
#include <iostream>

int main() {
    // run_visualization();
    // return 0;

    // Get the data from the csv file and split it into training and testing data
    std::vector<std::vector<Record>> data = getCsvData();

    // Create the neural network with 4 input nodes, 5 hidden nodes, and 3 output nodes
    NeuralNetwork nn(4, 5, 3);

    // The number of epochs to train the neural network for is equal to the number of training data points
    unsigned int NUM_TRAIN_EPOCHS = data[0].size();

    // Train the neural network with the training data for num train epochs given in the dataExtract.cpp file
    for (int i{}; i < NUM_TRAIN_EPOCHS; i++) {
        nn.train(data[0][i]);
    }   


    // Test the neural network with the testing data and print the accuracy of the network
    for(int i{}; i < data[1].size(); i++) {
        nn.test(data[1][i]);
    }

}

/* Run command in root directory of project
cmake --build build
.\build\Iris.exe
*/