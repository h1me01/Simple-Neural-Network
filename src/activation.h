#include <cmath>
#include <algorithm>

enum ActivationType
{
    LINEAR,
    SIGMOID,
    ReLU
};

float activate (float x, ActivationType type);

float activateDerivative (float x, ActivationType type);
