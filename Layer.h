#include <vector>

class Neuron;

class Layer
{
public:
    Layer(int neuronCount, int outputsPerNeuron);

    std::vector<Neuron> &getNeurons(void) { return m_neurons; }
    const std::vector<Neuron> &getNeurons() const { return m_neurons; }
    int getNeuronCount(void) const { return m_neurons.size(); }
    int getFunctionalNeuronCount(void) const { return m_neurons.size() - 1; } // last neuron is the bias neuron

    void setValues(const std::vector<double> &inputVals);
    std::vector<double> getValues() const;

    void feedForward(const Layer &prevLayer);

    double calcError(const std::vector<double> &targetVals) const;
    void calcGradients(const std::vector<double> &targetVals);
    void calcGradients(const Layer &nextLayer);
    void updateOutputWeights(const Layer &nextLayer);

private:
    std::vector<Neuron> m_neurons; 
};