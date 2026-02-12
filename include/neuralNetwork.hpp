#ifndef neuralNetwork_HPP
#define neuralNetwork_HPP

inline double sigmoid(double x);
double mean_squared_error(const double* prediction, const double* actual, int size);

class NeuralNetwork {
    public:
        NeuralNetwork(unsigned int inputNum, unsigned int outputNum, unsigned int HiddenLayerNum);



    private:
    

};


#endif