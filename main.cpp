#include "NeuralNetwork.h"
#include "TrainingData.h"
#include "Layer.h"
#include "Neuron.h"
#include "HyperParameters.h"

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip> 

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>
#include <QtWidgets/qboxlayout.h>

QT_CHARTS_USE_NAMESPACE

int main(int argc, char *argv[])
{
	TrainingData trainData("./iris.data");
	NeuralNetwork net(trainData.getTopology());

	QApplication a(argc, argv);

	// UI
	QLineSeries *series = new QLineSeries();
	QChart *chart = new QChart();
	chart->addSeries(series);
	chart->setTitle("Error drop rate");
	chart->createDefaultAxes();
	chart->setAnimationOptions(QChart::AllAnimations);
	chart->axisX()->setRange(0, HyperParameters::EPOCHS_COUNT);
	chart->axisY()->setRange(0, 0.6);

	QChartView *chartView1 = new QChartView(chart);
	chartView1->setRenderHint(QPainter::Antialiasing);

	QPieSeries *pie = new QPieSeries();
	pie->setHoleSize(0.35);
	QPieSlice *pass = pie->append("Pass", 0);
	QPieSlice *fail = pie->append("Fail", 0);
	
	QChartView *chartView2 = new QChartView();
	chartView2->setRenderHint(QPainter::Antialiasing);
	chartView2->chart()->setTitle("Test success");
	chartView2->chart()->addSeries(pie);
	chartView2->chart()->legend()->setAlignment(Qt::AlignBottom);

	QVBoxLayout *lay = new QVBoxLayout();
	lay->addWidget(chartView1);
	lay->addWidget(chartView2);
	QWidget *container = new QWidget();
	container->setLayout(lay);

	QMainWindow window;
	window.setCentralWidget(container);
	window.resize(600, 800);
	window.show();

	// ALGORITHM
	int epoch = 0;
	bool done = false;
	double minError = 1.0;
	while (epoch < HyperParameters::EPOCHS_COUNT && !done) {
		epoch++;
		std::cout << std::endl << "Epoch " << epoch;

		// TRAIN
		std::vector<TrainingSample> trainingSet = trainData.getTrainingSet(true);
		for (int i = 0; i < trainingSet.size(); ++i) {
			TrainingSample sample = trainingSet.at(i);
			net.feedForward(sample.features);
			net.backProp(sample.targetOutput);
		}

		// VALIDATE
		std::vector<TrainingSample> validationSet = trainData.getValidationSet();
		double errorSum = 0.0;
		for (int i = 0; i < validationSet.size(); ++i) {
			TrainingSample sample = trainingSet.at(i);
			net.feedForward(sample.features);
			double error = net.getError(sample.targetOutput);
			errorSum += error;
		}

		double epochError = errorSum / std::max(int(validationSet.size()), 1);
		std::cout << std::endl << "Epoch error: " << epochError << std::endl;
		if (epochError < 0.05) {
			//done = true; // need more testing to see if this is a good idea
		}

		series->append(QPointF(epoch, epochError));
		qApp->processEvents();
	}

	// TEST
	std::vector<TrainingSample> testSet = trainData.getTestSet();
	for (int i = 0; i < testSet.size(); ++i) {
		TrainingSample sample = testSet.at(i);
		net.feedForward(sample.features);

		std::vector<double> results = net.getResults();
		int maxIndex = 0;
		double currentMax = 0.0;
		for (int j = 0; j < results.size(); ++j) {
			if (results.at(j) > currentMax) {
				maxIndex = j;
				currentMax = results.at(j);
			}
		}

		if (sample.targetOutput.at(maxIndex) == 1.0) {
			pass->setValue(pass->value() + 1);
			pass->setLabel("Pass: " + QString::number(pass->value()));
		}
		else {
			fail->setValue(fail->value() + 1);
			fail->setLabel("Fail: " + QString::number(fail->value()));
		}
		qApp->processEvents();
	}

    std::cout << std::endl << "Done" << std::endl;
	return a.exec();
}