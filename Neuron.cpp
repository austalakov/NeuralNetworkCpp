#include "Neuron.h"
#include "Layer.h"
#include "HyperParameters.h"

Neuron::Neuron(int numOutputs)
{
    for (int c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(randomWeight());
        m_outputWeightDeltas.push_back(0.0);
    }
}

void Neuron::feedForward(double input)
{
    setOutputVal(Neuron::activationFunction(input));
}

void Neuron::updateGradient(double delta)
{
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::updateOutputWeight(int index, double otherGradient)
{
    double myContribution = HyperParameters::LEARNING_RATE * getOutputVal() * otherGradient;

    double oldDeltaWeight = m_outputWeightDeltas.at(index);
    double momentum = HyperParameters::MOMENTUM_FACTOR * oldDeltaWeight;

    double newDeltaWeight = myContribution + momentum;

    m_outputWeightDeltas[index] = newDeltaWeight;
    m_outputWeights[index] += newDeltaWeight;
}

double Neuron::activationFunction(double x)
{
    // logistic sigmoid
    // it could be replaced by tanh(x) because exp(x) is a bit slower (but it has bigger spread)
    return 1 / (1 + exp(-x));
}

double Neuron::activationFunctionDerivative(double x)
{
    // derivative
    return x * (1 - x);
}

double Neuron::randomWeight(void) 
{ 
    return rand() / double(RAND_MAX); 
}