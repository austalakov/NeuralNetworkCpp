#include "TrainingData.h"
#include "HyperParameters.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>

TrainingData::TrainingData(std::string file)
{
    int featureCount = 4;
    int targetsCount = 3;
	std::vector<TrainingSample> allSamples;

    std::ifstream infile(file);
    std::string line;
    while (getline(infile, line))
    {
        std::vector<double> features;
        std::vector<double> targetValues;

        std::istringstream iss(line);
        std::string s;    
        while (getline(iss, s, ',')) {
            if (s.find("Iris-") != std::string::npos) {
                if (s.find("setosa") != std::string::npos) {
                    targetValues.push_back(1.0);
                    targetValues.push_back(0.0);
                    targetValues.push_back(0.0);
                } else if (s.find("versicolor") != std::string::npos) {
                    targetValues.push_back(0.0);
                    targetValues.push_back(1.0);
                    targetValues.push_back(0.0);
                } else if (s.find("virginica") != std::string::npos) {
                    targetValues.push_back(0.0);
                    targetValues.push_back(0.0);
                    targetValues.push_back(1.0);
                }
            } else {
                double feature = stod(s);
                features.push_back(feature);
            }
        }

        TrainingSample sample;
        sample.features = features;
        sample.targetOutput = targetValues;
		allSamples.push_back(sample);
    }

    // shuffle
	std::srand(std::time(0));
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(allSamples), std::end(allSamples), rng);

	int trainingIndex = (allSamples.size() * 6) / 10; // 60% training data
	int validationIndex = trainingIndex + (allSamples.size() * 2) / 10; // 20% validation and 20% test

	std::copy(allSamples.begin(), allSamples.begin() + trainingIndex,
		std::back_inserter(m_training));

	std::copy(allSamples.begin() + trainingIndex, allSamples.begin() + validationIndex,
		std::back_inserter(m_validation));

	std::copy(allSamples.begin() + validationIndex, allSamples.end(),
		std::back_inserter(m_test));

    m_topology.push_back(featureCount);
	for (int i = 0; i < HyperParameters::HIDDEN_LAYER_COUNT; ++i) {
		m_topology.push_back(HyperParameters::HIDDEN_LAYER_NEURON_COUNT); // hidden layer
	}
    m_topology.push_back(targetsCount);
}

std::vector<TrainingSample> TrainingData::getTrainingSet(bool shuffle)
{ 
	if (shuffle)
	{
		std::srand(std::time(0));
		auto rng = std::default_random_engine{};
		std::shuffle(std::begin(m_training), std::end(m_training), rng);
	}
	return m_training; 
}