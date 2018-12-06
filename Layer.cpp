#include "Layer.h"
#include "Neuron.h"

Layer::Layer(int neuronCount, int outputsPerNeuron)
{
	if (outputsPerNeuron != 0) 
		outputsPerNeuron++; // need one more output for the next layer's bias neuron

    for (int i = 0; i < neuronCount; ++i) {
        m_neurons.push_back(Neuron(outputsPerNeuron));
    }

    Neuron biasNeuron(outputsPerNeuron);
    biasNeuron.setOutputVal(1.0);
    m_neurons.push_back(biasNeuron);
}

void Layer::setValues(const std::vector<double> &inputVals)
{
    for (int i = 0; i < inputVals.size(); ++i) {
        getNeurons().at(i).setOutputVal(inputVals.at(i));
    }
}

std::vector<double> Layer::getValues() const
{
    std::vector<double> result;

    for (int n = 0; n < getFunctionalNeuronCount(); ++n) {
        result.push_back(getNeurons().at(n).getOutputVal());
    }

    return result;
}

void Layer::feedForward(const Layer &prevLayer)
{
    for (int i = 0; i < getFunctionalNeuronCount(); ++i) {
        Neuron &myNeuron = getNeurons().at(i);
        double sum = 0.0;

        for (int n = 0; n < prevLayer.getNeuronCount(); ++n) { // including the bias neuron of prevLayer
            Neuron prevLayerNeuron = prevLayer.getNeurons().at(n);
            double outVal = prevLayerNeuron.getOutputVal();
            double weight = prevLayerNeuron.getOutputWeights().at(i);
            sum += outVal * weight;
        }

        myNeuron.feedForward(sum);
    }
}

double Layer::calcError(const std::vector<double> &targetVals) const
{
    double error = 0.0;

    for (int n = 0; n < getFunctionalNeuronCount(); ++n) {  
        double delta = targetVals.at(n) - getNeurons().at(n).getOutputVal();
        error += delta * delta;
    }
    error /= getFunctionalNeuronCount(); // get average error squared
    error = sqrt(error);

    return error;
}

void Layer::calcGradients(const std::vector<double> &targetVals)
{
    for (int n = 0; n < getFunctionalNeuronCount(); ++n) {
        Neuron &myNeuron = getNeurons().at(n);
        double targetVal = targetVals.at(n);
        double delta = targetVal - myNeuron.getOutputVal();
        myNeuron.updateGradient(delta);
    }
}

void Layer::calcGradients(const Layer &nextLayer)
{
    for (int i = 0; i < getNeuronCount(); ++i) { // including the bias neuron 
        Neuron &myNeuron = getNeurons().at(i);
        double sum = 0.0; // derivatives of weigths on next layer

        for (int n = 0; n < nextLayer.getFunctionalNeuronCount(); ++n) {
            double synapseWeight = myNeuron.getOutputWeights().at(n);
            double nextLayerNeuronGradient = nextLayer.getNeurons().at(n).getGradient();
            sum += synapseWeight * nextLayerNeuronGradient;
        }

        myNeuron.updateGradient(sum);
    }
}

void Layer::updateOutputWeights(const Layer &nextLayer)
{ 
    // update the weigths based on the gradients of the next layer
    for (int i = 0; i < getNeuronCount(); ++i) { // including the bias neuron 
        Neuron &myNeuron = getNeurons().at(i);

        for (int n = 0; n < nextLayer.getFunctionalNeuronCount(); ++n) {
            Neuron nextLayerNeuron = nextLayer.getNeurons().at(n);
            myNeuron.updateOutputWeight(n, nextLayerNeuron.getGradient());
        }   
    }
}