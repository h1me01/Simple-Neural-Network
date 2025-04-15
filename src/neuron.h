#pragma once

#include "types.h"
#include <iostream>

#include <random>

inline void heInitialize(Array &weights, int size)
{
    // std::random_device rd;

    // use rd() if you want random numbers every time
    // this is not always optimal tough (especially when testing), since it is possible
    // that the networks doesn't learn if the random numbers are bad
    // if the network complexity is big, something like that is very unlikely
    // but with our simple network, it is actually really common
    std::mt19937 gen(42);

    float scale = sqrt(2.0f / size);
    std::normal_distribution<float> dist(0, scale);

    for (int i = 0; i < size; i++)
        weights[i] = dist(gen);
}

struct Neuron
{
    float bias, bias_gradient;
    Array weights, weight_gradients;

    Neuron() {}

    Neuron(int size)
    {
        weights = Array(size);
        weight_gradients = Array(size);
        heInitialize(weights, size);

        bias = 0;
        bias_gradient = 0;
    }

    float dot(Array &input)
    {
        assert(input.getSize() == weights.getSize());

        float sum = bias;
        for (int i = 0; i < weights.getSize(); i++)
            sum += weights[i] * input[i];

        return sum;
    }

    void update(float lr)
    {
        for (int i = 0; i < weights.getSize(); i++)
        {
            weights[i] -= lr * weight_gradients[i];
            weight_gradients[i] = 0;
        }

        bias -= lr * bias_gradient;
        bias_gradient = 0;
    }
};
