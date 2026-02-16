#include "visualizer.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

void run_visualization() {
    sf::RenderWindow window(sf::VideoMode({800, 600}), "Neural Network");

    float radius = 20.f;
    sf::CircleShape neuron(radius);
    neuron.setFillColor(sf::Color::Transparent);
    neuron.setOutlineColor(sf::Color::Cyan);
    neuron.setOutlineThickness(2.f);
    
    neuron.setOrigin({radius, radius});

    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) window.close();
        }

        window.clear(sf::Color::Black);

        for (int i = 0; i < 3; ++i) {
            neuron.setPosition({100.f, 150.f + (i * 150.f)});
            window.draw(neuron);
        }

        window.display();
    }
}