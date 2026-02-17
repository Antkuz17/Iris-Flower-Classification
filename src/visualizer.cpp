#include "visualizer.hpp"
#include <SFML/Graphics.hpp>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>

// ─── Constants ───────────────────────────────────────────────────────────────
static const sf::Color BG_COLOR        = {30,  30,  30};
static const sf::Color CARD_COLOR      = {45,  45,  45};
static const sf::Color OUTLINE_COLOR   = {80,  80,  80};
static const sf::Color WHITE           = sf::Color::White;
static const sf::Color ORANGE          = {255, 140, 0};
static const sf::Color ORANGE_DIM      = {180, 100, 0};
static const sf::Color TEXT_DIM        = {160, 160, 160};

static const float NODE_RADIUS   = 28.f;
static const float WINDOW_W      = 1100.f;
static const float WINDOW_H      = 700.f;

// ─── Draw a clock-dial node ──────────────────────────────────────────────────
// activation in [0,1] maps hand angle from -135deg to +135deg
static void drawDialNode(sf::RenderWindow& window, sf::Vector2f pos, float activation, sf::Color rimColor)
{
    // Background circle
    sf::CircleShape bg(NODE_RADIUS);
    bg.setOrigin({NODE_RADIUS, NODE_RADIUS});
    bg.setPosition(pos);
    bg.setFillColor(CARD_COLOR);
    bg.setOutlineColor(rimColor);
    bg.setOutlineThickness(2.f);
    window.draw(bg);

    // Clock hand
    float angle_deg = -135.f + activation * 270.f;   // -135 → +135
    float angle_rad = angle_deg * (3.14159265f / 180.f);
    float handLen   = NODE_RADIUS * 0.65f;
    sf::Vector2f tip = {
        pos.x + handLen * std::sin(angle_rad),
        pos.y - handLen * std::cos(angle_rad)
    };

    sf::Vertex hand[2] = {
        sf::Vertex{pos,  ORANGE},
        sf::Vertex{tip,  ORANGE}
    };
    window.draw(hand, 2, sf::PrimitiveType::Lines);

    // Center dot
    sf::CircleShape dot(3.f);
    dot.setOrigin({3.f, 3.f});
    dot.setPosition(pos);
    dot.setFillColor(ORANGE);
    window.draw(dot);
}

// ─── Draw a button ───────────────────────────────────────────────────────────
struct Button {
    sf::RectangleShape shape;
    sf::Text           label;
    bool               enabled = true;

    Button(sf::Vector2f pos, sf::Vector2f size, const std::string& text, const sf::Font& font)
        : label(font, text, 18)
    {
        shape.setSize(size);
        shape.setPosition(pos);
        shape.setFillColor(CARD_COLOR);
        shape.setOutlineColor(OUTLINE_COLOR);
        shape.setOutlineThickness(2.f);

        sf::FloatRect tb = label.getLocalBounds();
        label.setOrigin({tb.position.x + tb.size.x / 2.f, tb.position.y + tb.size.y / 2.f});
        label.setPosition({pos.x + size.x / 2.f, pos.y + size.y / 2.f});
        label.setFillColor(WHITE);
    }

    bool contains(sf::Vector2f p) const { return shape.getGlobalBounds().contains(p); }

    void draw(sf::RenderWindow& w) {
        shape.setFillColor(enabled ? CARD_COLOR : sf::Color{35,35,35});
        label.setFillColor(enabled ? WHITE : TEXT_DIM);
        w.draw(shape);
        w.draw(label);
    }
};

// ─── Main visualizer ─────────────────────────────────────────────────────────
void run_visualization(NeuralNetwork& nn, const std::vector<std::vector<Record>>& data)
{
    sf::RenderWindow window(sf::VideoMode({(unsigned)WINDOW_W, (unsigned)WINDOW_H}),
                            "Neural Network Visualizer");
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.openFromFile("C:/Windows/Fonts/segoeui.ttf")) {
        font.openFromFile("C:/Windows/Fonts/arial.ttf");
    }

    // ── Layer x positions ──
    // Input(4) → Hidden(5) → Output(3)
    const float x_input  = 200.f;
    const float x_hidden = 550.f;
    const float x_output = 900.f;

    auto layerPositions = [](float x, int n) -> std::vector<sf::Vector2f> {
        std::vector<sf::Vector2f> pts;
        float spacing = std::min(120.f, (WINDOW_H - 120.f) / (n + 1));
        float total   = spacing * (n - 1);
        float startY  = WINDOW_H / 2.f - total / 2.f;
        for (int i = 0; i < n; ++i)
            pts.push_back({x, startY + i * spacing});
        return pts;
    };

    auto inputPos  = layerPositions(x_input,  4);
    auto hiddenPos = layerPositions(x_hidden, 5);
    auto outputPos = layerPositions(x_output, 3);

    // ── Buttons ──
    Button trainBtn({30.f,  30.f}, {120.f, 44.f}, "TRAIN",  font);
    Button testBtn ({30.f,  90.f}, {120.f, 44.f}, "TEST",   font);
    testBtn.enabled = false;

    // ── State ──
    bool  training     = false;
    bool  trained      = false;
    int   currentEpoch = 0;
    int   totalEpochs  = 1000;
    int   epochStep    = 100;   // update visual every 100 epochs
    float lastCost     = 0.f;
    std::string statusText = "Press TRAIN to start";

    // Activations (displayed on dials)
    std::vector<float> act_input (4, 0.5f);
    std::vector<float> act_hidden(5, 0.5f);
    std::vector<float> act_output(3, 0.5f);

    // Cache weights for connection drawing (flattened)
    // W1: 4x5,  W2: 5x3
    auto getWeights = [&](const Matrix& m) {
        std::vector<float> w;
        for (unsigned r = 0; r < m.get_num_rows(); ++r)
            for (unsigned c = 0; c < m.get_num_col(); ++c)
                w.push_back((float)m.get_val(r, c));
        return w;
    };

    std::vector<float> w1 = getWeights(nn.getW1());
    std::vector<float> w2 = getWeights(nn.getW2());

    // Helper: run one sample forward and pull activations
    auto sampleActivations = [&](int sampleIdx) {
        const auto& recs = data[0];
        int idx = sampleIdx % (int)recs.size();
        Matrix X(1, 4, 0.0);
        X.set_val(0, 0, recs[idx].sepal_length);
        X.set_val(0, 1, recs[idx].sepal_width);
        X.set_val(0, 2, recs[idx].pedal_length);
        X.set_val(0, 3, recs[idx].pedal_width);

        act_input[0] = (float)recs[idx].sepal_length;
        act_input[1] = (float)recs[idx].sepal_width;
        act_input[2] = (float)recs[idx].pedal_length;
        act_input[3] = (float)recs[idx].pedal_width;

        nn.forward_propagation(X);

        for (int i = 0; i < 5; ++i)
            act_hidden[i] = (float)nn.getA1().get_val(0, i);
        Matrix A2 = nn.forward_propagation(X);
        for (int i = 0; i < 3; ++i)
            act_output[i] = (float)A2.get_val(0, i);
    };

    // Status text widget
    sf::Text statusLabel(font, statusText, 15);
    statusLabel.setFillColor(TEXT_DIM);
    statusLabel.setPosition({30.f, 150.f});

    sf::Text epochLabel(font, "", 15);
    epochLabel.setFillColor(ORANGE);
    epochLabel.setPosition({30.f, 175.f});

    sf::Text costLabel(font, "", 15);
    costLabel.setFillColor(TEXT_DIM);
    costLabel.setPosition({30.f, 200.f});

    // Layer labels
    auto makeLayerLabel = [&](const std::string& t, float x) {
        sf::Text lbl(font, t, 14);
        lbl.setFillColor(TEXT_DIM);
        sf::FloatRect b = lbl.getLocalBounds();
        lbl.setOrigin({b.position.x + b.size.x / 2.f, 0.f});
        lbl.setPosition({x, 60.f});
        return lbl;
    };
    auto lblInput  = makeLayerLabel("Input (4)",   x_input);
    auto lblHidden = makeLayerLabel("Hidden (5)",  x_hidden);
    auto lblOutput = makeLayerLabel("Output (3)",  x_output);

    // Output class names
    const char* classNames[3] = {"Setosa", "Versicolor", "Virginica"};

    // ── Connection draw helper ────────────────────────────────────────────────
    auto drawConnections = [&](
        const std::vector<sf::Vector2f>& from,
        const std::vector<sf::Vector2f>& to,
        const std::vector<float>& weights)
    {
        // Find weight range for normalization
        float wMin = weights[0], wMax = weights[0];
        for (float w : weights) { wMin = std::min(wMin, w); wMax = std::max(wMax, w); }
        float wRange = wMax - wMin;
        if (wRange < 1e-6f) wRange = 1.f;

        int idx = 0;
        for (int f = 0; f < (int)from.size(); ++f) {
            for (int t = 0; t < (int)to.size(); ++t, ++idx) {
                float norm = (weights[idx] - wMin) / wRange;  // 0..1
                uint8_t brightness = (uint8_t)(40 + norm * 160);
                float   thickness  = 0.5f + norm * 2.5f;

                // Positive weights → orangey, negative → blueish
                sf::Color col;
                if (weights[idx] >= 0)
                    col = {brightness, (uint8_t)(brightness / 2), 0, 180};
                else
                    col = {0, (uint8_t)(brightness / 2), brightness, 180};

                // Draw as a thin rectangle (line with thickness)
                sf::Vector2f dir = to[t] - from[f];
                float len = std::sqrt(dir.x*dir.x + dir.y*dir.y);
                float angle = std::atan2(dir.y, dir.x) * 180.f / 3.14159265f;

                sf::RectangleShape line({len, thickness});
                line.setOrigin({0.f, thickness / 2.f});
                line.setPosition(from[f]);
                line.setRotation(sf::degrees(angle));
                line.setFillColor(col);
                window.draw(line);
            }
        }
    };

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (window.isOpen())
    {
        // ── Events ──
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) window.close();

            if (const auto* mp = event->getIf<sf::Event::MouseButtonPressed>()) {
                sf::Vector2f mpos = {(float)mp->position.x, (float)mp->position.y};

                if (trainBtn.contains(mpos) && trainBtn.enabled) {
                    training     = true;
                    currentEpoch = 0;
                    trainBtn.enabled = false;
                    statusText = "Training...";
                }

                if (testBtn.contains(mpos) && testBtn.enabled) {
                    // Run test and show result
                    int correct = 0;
                    const auto& recs = data[1];
                    for (int i = 0; i < (int)recs.size(); ++i) {
                        Matrix X(1, 4, 0.0);
                        X.set_val(0, 0, recs[i].sepal_length);
                        X.set_val(0, 1, recs[i].sepal_width);
                        X.set_val(0, 2, recs[i].pedal_length);
                        X.set_val(0, 3, recs[i].pedal_width);
                        Matrix A2 = nn.forward_propagation(X);

                        int pred = 0, actual = 0;
                        for (int j = 1; j < 3; ++j) {
                            if (A2.get_val(0,j) > A2.get_val(0,pred)) pred = j;
                            Matrix Y(1, 3, 0.0);
                            Y.set_val(0, 0, recs[i].one_hot[0]);
                            Y.set_val(0, 1, recs[i].one_hot[1]);
                            Y.set_val(0, 2, recs[i].one_hot[2]);
                            if (Y.get_val(0,j) > Y.get_val(0,actual)) actual = j;
                        }
                        if (pred == actual) correct++;

                        // Show last sample activations
                        if (i == (int)recs.size() - 1) {
                            act_input[0] = (float)recs[i].sepal_length;
                            act_input[1] = (float)recs[i].sepal_width;
                            act_input[2] = (float)recs[i].pedal_length;
                            act_input[3] = (float)recs[i].pedal_width;
                            for (int k = 0; k < 5; ++k)
                                act_hidden[k] = (float)nn.getA1().get_val(0, k);
                            for (int k = 0; k < 3; ++k)
                                act_output[k] = (float)A2.get_val(0, k);
                        }
                    }
                    float acc = (float)correct / recs.size() * 100.f;
                    statusText = "Test accuracy: " + std::to_string((int)acc) + "%";
                }
            }
        }

        // ── Training step ──
        if (training && currentEpoch < totalEpochs) {
            // Train epochStep epochs
            const auto& recs = data[0];
            double totalCost = 0.0;
            for (int e = 0; e < epochStep && currentEpoch < totalEpochs; ++e, ++currentEpoch) {
                for (int i = 0; i < (int)recs.size(); ++i) {
                    Matrix X(1, 4, 0.0);
                    X.set_val(0, 0, recs[i].sepal_length);
                    X.set_val(0, 1, recs[i].sepal_width);
                    X.set_val(0, 2, recs[i].pedal_length);
                    X.set_val(0, 3, recs[i].pedal_width);

                    Matrix Y(1, 3, 0.0);
                    Y.set_val(0, 0, recs[i].one_hot[0]);
                    Y.set_val(0, 1, recs[i].one_hot[1]);
                    Y.set_val(0, 2, recs[i].one_hot[2]);

                    Matrix A2 = nn.forward_propagation(X);
                    totalCost += mean_squared_error(A2, Y);
                    GradientStruct g = nn.back_propagation(X, Y);
                    nn.update_weights(g, 0.1);
                }
            }
            lastCost = (float)(totalCost / (epochStep * recs.size()));

            // Update weight cache
            w1 = getWeights(nn.getW1());
            w2 = getWeights(nn.getW2());

            // Update activations from sample 0
            sampleActivations(currentEpoch);

            if (currentEpoch >= totalEpochs) {
                training = false;
                trained  = true;
                testBtn.enabled  = true;
                trainBtn.enabled = false;
                statusText = "Training complete!";
            }
        }

        // ── Update labels ──
        statusLabel.setString(statusText);
        epochLabel.setString(training || trained ? "Epoch: " + std::to_string(currentEpoch) + "/" + std::to_string(totalEpochs) : "");
        costLabel.setString(lastCost > 0.f ? "Cost: " + std::to_string(lastCost).substr(0, 8) : "");

        // ── Draw ──
        window.clear(BG_COLOR);

        // Connections (drawn behind nodes)
        drawConnections(inputPos, hiddenPos, w1);
        drawConnections(hiddenPos, outputPos, w2);

        // Input nodes
        for (int i = 0; i < 4; ++i)
            drawDialNode(window, inputPos[i], act_input[i], OUTLINE_COLOR);

        // Hidden nodes
        for (int i = 0; i < 5; ++i)
            drawDialNode(window, hiddenPos[i], act_hidden[i], OUTLINE_COLOR);

        // Output nodes
        for (int i = 0; i < 3; ++i) {
            // Highlight the winning output in orange
            sf::Color rim = OUTLINE_COLOR;
            if (trained || !training) {
                float mx = *std::max_element(act_output.begin(), act_output.end());
                if (act_output[i] == mx) rim = ORANGE;
            }
            drawDialNode(window, outputPos[i], act_output[i], rim);

            // Class label
            sf::Text clbl(font, classNames[i], 13);
            clbl.setFillColor(TEXT_DIM);
            clbl.setPosition({outputPos[i].x + NODE_RADIUS + 8.f, outputPos[i].y - 8.f});
            window.draw(clbl);
        }

        // Layer labels
        window.draw(lblInput);
        window.draw(lblHidden);
        window.draw(lblOutput);

        // Buttons
        trainBtn.draw(window);
        testBtn.draw(window);

        // Status / epoch / cost
        window.draw(statusLabel);
        window.draw(epochLabel);
        window.draw(costLabel);

        window.display();
    }
}