#include <vector>

class Neuron
{
public:
    Neuron(int numOutputs);

    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    double getGradient(void) const { return m_gradient; }
    const std::vector<double> &getOutputWeights() const { return m_outputWeights; }

    void feedForward(double input);
    void updateGradient(double delta);
    void updateOutputWeight(int index, double otherGradient);

private:
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void);

    double m_outputVal;
    std::vector<double> m_outputWeights;
    std::vector<double> m_outputWeightDeltas;
    double m_gradient;
};