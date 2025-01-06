#include <cmath>

inline float mse(float prediction, float target)
{
    float diff = prediction - target;
    return diff * diff;
}

inline float mseDerivative(float prediction, float target)
{
    return 2 * (prediction - target);
}