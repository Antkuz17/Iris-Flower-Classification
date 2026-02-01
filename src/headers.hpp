#ifndef DATA_EXTRACT_HPP
#define DATA_EXTRACT_HPP

#include <vector>
#include <string>

struct Record {
    double pedal_length;
    double pedal_width;
    double sepal_length;
    double sepal_width;
    std::string flower_type;
    double one_hot[3];
};

std::vector<std::vector<Record>> getCsvData();
std::vector<std::vector<Record>> splitData(int trainNum, int testNum, Record* dataPoints);
void shuffleVector(Record* dataPoints, int size);

double normalize_pedal_length(double value);
double normalize_pedal_width(double value);
double normalize_sepal_length(double value);
double normalize_sepal_width(double value);


inline double sigmoid(double x);

#endif