# NeuralNetworkCpp

## What it is
An implementation of a Neural Network in C++ and its application to the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)
The Neural Netowrk itself is in pure C++11 but the visualization of the results uses Qt. Below are the results of running it once.

![results](https://github.com/austalakov/NeuralNetworkCpp/blob/master/results.png)

## How to use it
Download the latest release from [here](https://github.com/austalakov/NeuralNetworkCpp/releases) and run the executable.
At the moment only a Windows x64 version is available

## How to compile it
The source is available as a Visual Studio project and as a QtCreator .pro file.
Either of those IDEs will be able to build it by opening the corresponding project file.
In addition, the code should compile easily under GCC on Linux, but it has not been tested yet.

## Reflection
I chose to write the netwrok in C++ using Object-Oriented design rather than the more popular matrix-manipulation in languages like Python. This is just s personal preference, as I am more comfortable with the OO paradigm.
The network definitely has some issues and is a bit unreliable - it produces good results most of the time it is ran, but there are cases when it gives close to 50% fail rate on the test set.
In addition, the error rate seems to always have spikes, so it seems it might be falling victim to overtuning.
Tweaking with the hyper parameters might help, but I would need to spend more time on it.
