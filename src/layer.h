#pragma once

#include "loss.h"
#include "neuron.h"
#include "activation.h"

#include <vector>

class Layer
{
public:
    Layer(int input_size, int output_size, Layer *next_layer, ActivationType activation_type)
        : input_size(input_size), size(output_size), next_layer(next_layer), activation_type(activation_type)
    {
        activation = Array(output_size);
        deltas = Array(output_size);
        input = Array(output_size);

        neurons.resize(output_size);
        for (int i = 0; i < output_size; i++)
            neurons[i] = Neuron(input_size);
    }

    void forward(Array input)
    {
        this->input = input;

        for (int i = 0; i < size; i++)
            activation[i] = activate(neurons[i].dot(input), activation_type);
    }

    // target value will only be used in the output layer
    void backward(Array targets = Array())
    {
        for (int i = 0; i < size; i++)
        {
            if (next_layer == nullptr)
            {
                // output layer
                deltas[i] = mseDerivative(activation[i], targets[i]) * activateDerivative(activation[i], activation_type);
            }
            else
            {
                // hidden layer 
                const std::vector<Neuron> &next_neurons = next_layer->getNeurons();
                const Array &next_deltas = next_layer->getDeltas();
                // use the chain rule to calculate the gradients
                deltas[i] = 0;
                for (size_t j = 0; j < next_neurons.size(); j++)
                    deltas[i] += next_neurons[j].weights[i] * next_deltas[j];
                deltas[i] *= activateDerivative(activation[i], activation_type);
            }

            // add the gradients to each weight
            // make sure to use input here to determine which weights are important
            for (int j = 0; j < input_size; j++)
                neurons[i].weight_gradients[j] += deltas[i] * input[j];
            // since the bias has not a direct connection to the input, we just at the delta
            neurons[i].bias_gradient += deltas[i];
        }
    }

    void update(float lr)
    {
        for (int i = 0; i < size; i++)
            neurons[i].update(lr);
    }

    int getSize() { return size; }

    int getInputSize() { return input_size; }

    Array getActivation() { return activation; }

    Array getDeltas() { return deltas; }

    std::vector<Neuron> getNeurons() { return neurons; }

private:
    int input_size, size;
    Layer *next_layer;
    ActivationType activation_type;
    Array activation, deltas, input;
    std::vector<Neuron> neurons;
};
