#include <vector>

class Layer;

class NeuralNetwork
{
public:
    NeuralNetwork(const std::vector<int> &topology);
    // NeuralNetwork(string fileName) // deserialize to set topology and weights
    // void serialize(string fileName) // serialize topology and weights

    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);

    std::vector<double>  getResults() const;
	double getError(const std::vector<double> &targetVals) const;

private:
    std::vector<Layer> m_layers;
};