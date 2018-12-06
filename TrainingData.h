#include <vector>
#include <string>

struct TrainingSample
{
    std::vector<double> features;
    std::vector<double> targetOutput;
};

class TrainingData
{
public:
    TrainingData(std::string file);

	std::vector<TrainingSample> getTrainingSet(bool shuffle);
	std::vector<TrainingSample> getValidationSet() const { return m_validation; }
	std::vector<TrainingSample> getTestSet() const { return m_test; }
    std::vector<int> getTopology() const { return m_topology; }

private:
    std::vector<TrainingSample> m_training;
	std::vector<TrainingSample> m_validation;
	std::vector<TrainingSample> m_test;
    std::vector<int> m_topology;
};