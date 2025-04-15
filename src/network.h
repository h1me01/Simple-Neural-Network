#pragma once

#include "layer.h"

#include <iostream>
#include <memory>

// learning rate to control how fast the network should learn
const float LR = 0.01f;

struct NetInput
{
    Array input;
    Array target;

    NetInput(Array input, Array target)
        : input(input), target(target) {}
};

class Network
{
public:
    Network()
    {
        // build architecture here
        // output layer: has 4 input neurons, 1 output neuron, no next layer
        std::unique_ptr<Layer> output_layer = std::make_unique<Layer>(4, 1, nullptr, SIGMOID);
        // hidden layer 2: has 4 input neurons and 4 output neurons, output layer is the next
        std::unique_ptr<Layer> hidden_layer2 = std::make_unique<Layer>(4, 4, output_layer.get(), ReLU);
        // hidden layer 1: has two input neurons (input neurons must match the input data) and 4 output neurons, hidden 1 is next
        std::unique_ptr<Layer> hidden_layer1 = std::make_unique<Layer>(2, 4, hidden_layer2.get(), ReLU);

        layers.emplace_back(std::move(hidden_layer1));
        layers.emplace_back(std::move(hidden_layer2));
        layers.emplace_back(std::move(output_layer));
    }

    Array forward(Array input)
    {
        assert(input.getSize() == layers[0]->getInputSize());
        // first feed the first layer with the input
        layers[0]->forward(input);
        // after that you can feed the activation to the next layers
        for (size_t i = 1; i < layers.size(); i++)
            layers[i]->forward(layers[i - 1]->getActivation());

        return layers.back()->getActivation(); 
    }

    void backward(Array targets)
    {
        // first compute the errors of the output layer
        layers.back()->backward(targets);
        // then compute the rest of the hidden layers
        for (int i = layers.size() - 2; i >= 0; i--)
            layers[i]->backward(); 
    }

    void update()
    {
        // update the whole networks weights and biases
        for (size_t i = 0; i < layers.size(); i++)
            layers[i]->update(LR);
    }

    void train(const std::vector<NetInput> data, const int epochs)
    {
        std::cout << "Training network with " << data.size() << " datapoints." << std::endl;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            // measures how well the network is doing
            // the smaller, the better
            float total_loss = 0;

            for (const auto &sample : data)
            {
                Array prediction = forward(sample.input);
                
                backward(sample.target);
                update();

                for (int i = 0; i < prediction.getSize(); i++)
                    total_loss += mse(prediction[i], sample.target[i]);             
            }
            // make sure to get the mean
            total_loss /= data.size();

            if (epoch % (epochs / 50) == 0)
                std::cout << "epoch: " << epoch << " current loss: " << total_loss << std::endl;
        }
    }

private:
    std::vector<std::unique_ptr<Layer>> layers;
};
