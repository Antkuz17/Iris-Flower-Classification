#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>



// The struct used to store each data point from the csv file
struct Record {
    double pedal_length;
    double pedal_width;
    double sepal_length;
    double sepal_width;
    std::string flower_type;
};


int main () {
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

        
        
        listDataPoints[currentLine].pedal_length = std::stod(tokens[0]);
        listDataPoints[currentLine].pedal_width = std::stod(tokens[1]);
        listDataPoints[currentLine].sepal_length = std::stod(tokens[2]);
        listDataPoints[currentLine].sepal_width = std::stod(tokens[3]);

        
        currentLine++;
        
    }
    std::cout << currentLine;

    file.close();
}


// g++ dataExtract.cpp -o testing ; ./testing
