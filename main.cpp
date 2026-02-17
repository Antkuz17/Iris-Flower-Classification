
#include "visualizer.hpp"
#include "matrix.hpp"
#include "neuralNetwork.hpp"
#include "dataExtract.hpp"

int main() {
    auto data = getCsvData();
    NeuralNetwork nn(4, 5, 3);
    run_visualization(nn, data);
    return 0;
}

/* Run command in root directory of project
cmake --build build
.\build\Iris.exe
*/