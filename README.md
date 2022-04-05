# Diagnosing Pneumonia from X-Ray Images Using Convolutional Neural Network
## Introduction
Deep learning, also known as neural network, has gained traction in recent years, and is now being widely used across multiple fields, with one of its popular applications being image classification which uses the convolutional neural network (“CNN”). The potential of image classification is also being explored in the field of medical diagnosis. In this article, I discuss how we can build a convolutional neural network that would help us diagnose pneumonia from chest x-ray images.
## Neural Network
First, a neural network can be described as a series of algorithms that has an objective to recognise relationships or patterns in the data. A simple neural network consists of different types of layers, namely the input layer, the output layer and the hidden layers. Each layer contains a number of neurons, and each neuron is made up of algorithms that take the outputs from the previous layer as inputs to generate further outputs and pass them onto the next layer.

The below diagram shows an example architecture of a simple neural network. In this example, there are 4 variables in the input layer represented by the green circles. They are entered into each of the 4 neurons represented by the blue circles in the 1st hidden layer. These neurons would then generate further outputs to be passed onto the neurons in the 2nd hidden layer. Lastly, the outputs of the 2nd hidden layer would go into the neuron in the output layer represented by the orange circle which would generate the final output.

![simple neural network](https://github.com/tonytsoi/CNN_xray/blob/main/charts/simple%20neural%20network.jpg?raw=true)
