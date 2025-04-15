#include "activation.h"

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// assume x is already sigmoided
float sigmoidDerivative(float x)
{
    return x * (1.0f - x);
}

float relu(float x)
{
    return std::max(0.0f, x);
}

float reluDerivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

float activate(float x, ActivationType type)
{
    switch (type)
    {
    case SIGMOID:
        return sigmoid(x);
    case ReLU:
        return relu(x);
    default:
        return x; // must be linear
    }
}

float activateDerivative(float x, ActivationType type)
{
    switch (type)
    {
    case SIGMOID:
        return sigmoidDerivative(x);
    case ReLU:
        return reluDerivative(x);
    default:
        return 1.0f; // must be linear
    }
}
