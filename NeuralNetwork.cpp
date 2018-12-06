#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology)
{
    for (int i = 0; i < topology.size(); ++i) {
        int numOutputs = (i == topology.size() - 1) ? 0 : topology[i + 1];
        m_layers.push_back(Layer(topology.at(i), numOutputs));
    }
}

void NeuralNetwork::backProp(const std::vector<double> &targetVals)
{
    Layer &outputLayer = m_layers.back();

    // start back propagation by calculating gradients
    outputLayer.calcGradients(targetVals);

    for (int layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) { // hidden layers only
        Layer &currentLayer = m_layers.at(layerNum);
        Layer &nextLayer = m_layers.at(layerNum + 1);
        currentLayer.calcGradients(nextLayer);
    }

    // continue by updating connection weights
    for (int layerNum = m_layers.size() - 2; layerNum >= 0; --layerNum) { // hidden and input layers
        Layer &currentLayer = m_layers.at(layerNum);
        Layer &nextLayer = m_layers.at(layerNum + 1);
        currentLayer.updateOutputWeights(nextLayer);
    }
}

void NeuralNetwork::feedForward(const std::vector<double> &inputVals)
{
    // input layer
    m_layers.front().setValues(inputVals);

    // all other layers
    for (int i = 1; i < m_layers.size(); ++i) {
        Layer &prevLayer = m_layers.at(i - 1);
        m_layers.at(i).feedForward(prevLayer);
    }
}

std::vector<double> NeuralNetwork::getResults() const
{
    return m_layers.back().getValues();
}

double NeuralNetwork::getError(const std::vector<double> &targetVals) const
{
	return m_layers.back().calcError(targetVals);
}