#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>
#include "headers.hpp"


std::vector<std::vector<Record>> getCsvData () {
    std::ifstream file("../data/iris.data");

    // An array of all the the data point objects
    Record * listDataPoints = new Record[150];

    std::string line;

    // Tracks the current write entry of the array
    unsigned int currentLine{};

    // Until we hit the end of the file, access each line and populate the struct
    while(std::getline(file, line)){
        
        // Debugging line
        std::cout << line << std::endl;

        std::stringstream ss(line);

        std::string token;
        std::vector<std::string> tokens;

        while(std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        
        
    listDataPoints[currentLine].sepal_length = normalize_sepal_length(std::stod(tokens[0])); 
    listDataPoints[currentLine].sepal_width = normalize_sepal_width(std::stod(tokens[1]));  
    listDataPoints[currentLine].pedal_length = normalize_pedal_length(std::stod(tokens[2])); 
    listDataPoints[currentLine].pedal_width = normalize_pedal_width(std::stod(tokens[3]));  
    listDataPoints[currentLine].flower_type = tokens[4]; 
    
    if (listDataPoints[currentLine].flower_type == "Iris-setosa") {
        listDataPoints[currentLine].one_hot[0] = 1.0;
        listDataPoints[currentLine].one_hot[1] = 0.0;
        listDataPoints[currentLine].one_hot[2] = 0.0;
    } else if (listDataPoints[currentLine].flower_type == "Iris-versicolor") {
        listDataPoints[currentLine].one_hot[0] = 0.0;
        listDataPoints[currentLine].one_hot[1] = 1.0;
        listDataPoints[currentLine].one_hot[2] = 0.0;
    } else if (listDataPoints[currentLine].flower_type == "Iris-virginica") {
        listDataPoints[currentLine].one_hot[0] = 0.0;
        listDataPoints[currentLine].one_hot[1] = 0.0;
        listDataPoints[currentLine].one_hot[2] = 1.0;
    }

    currentLine++;
        
    }

    file.close();

    // The default iris dataset is ordered, so we need to shuffle it before splitting it into training and testing data
    shuffleVector(listDataPoints, 150);

    std::vector<std::vector<Record>> result = splitData(120, 30, listDataPoints);

    // Free the heap memory
    delete[] listDataPoints;

    // Return the vector containing the training and testing data
    return result;
}

// Returns a vector containing two vectors, the first being the training data and the second being the testing data
std::vector<std::vector<Record>> splitData(int trainNum, int testNum, Record* dataPoints) {

    // The vector is going to hold the training data vector at index 0 and the testing data vector at index 1
    std::vector<std::vector<Record>> array(2);

    array[0] = std::vector<Record>(trainNum);
    for(int i{}; i < trainNum; i++) {
        array[0][i] = dataPoints[i];
    }

    array[1] = std::vector<Record>(testNum);
    for(int i{}; i < testNum; i++) {
        array[1][i] = dataPoints[i + trainNum];
    }

    return array;
}   

// Shuffles the data points in the array since be default the Iris data is structured in an ordered way
void shuffleVector(Record * dataPoints, int size) {

    // 42 for the memes and the funnies 
    std::mt19937 seedValue(42);

    // Shuffle the vector
    std::shuffle(dataPoints, dataPoints + size, seedValue);
}


double normalize_pedal_length(double value){
    return ((value - 1.0) / 5.9);
}

double normalize_pedal_width(double value){
    return ((value - 0.1) / 2.4);
}

double normalize_sepal_length(double value){
    return ((value - 4.3) / 3.6);
}

double normalize_sepal_width(double value){
    return ((value - 2.0) / 2.4);
}

// g++ dataExtract.cpp -o testing ; ./testing

