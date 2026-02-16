cmake_minimum_required(VERSION 3.10)
project(IrisClassification)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(Iris 
    main.cpp 
    src/dataExtract.cpp 
    src/matrix.cpp 
    src/neuralNetwork.cpp
)

set_target_properties(Iris PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")