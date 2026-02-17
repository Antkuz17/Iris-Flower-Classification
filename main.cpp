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

    auto data = getCsvData();

    NeuralNetwork nn(4, 5, 3);

    nn.train(data, 1000, 0.1);
    nn.test(data);

}

/* Run command in root directory of project
cmake --build build
.\build\Iris.exe
*/