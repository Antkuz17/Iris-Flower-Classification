cmake_minimum_required(VERSION 3.10)
project(IrisClassification)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
include_directories("C:/Program Files/SFML-3.0.2/include")
link_directories("C:/Program Files/SFML-3.0.2/lib")

set(SFML_DIR "C:/Program Files/SFML-3.0.2/lib/cmake/SFML")

find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

add_executable(Iris 
    src/visualizer.cpp
    main.cpp 
    src/dataExtract.cpp 
    src/matrix.cpp 
    src/neuralNetwork.cpp
)

target_link_libraries(Iris PRIVATE sfml-graphics sfml-window sfml-system)


