#include "visualizer.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

void run_visualization() {
    std::cout << "Creating window..." << std::endl;
    
    sf::RenderWindow window(sf::VideoMode({800, 600}), "Neural Network");
    
    std::cout << "Window created" << std::endl;
    
    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                std::cout << "Close event received" << std::endl;
                window.close();
            }
        }
        
        window.clear(sf::Color::Black);
        window.display();
    }
    
    std::cout << "Window closed" << std::endl;
}