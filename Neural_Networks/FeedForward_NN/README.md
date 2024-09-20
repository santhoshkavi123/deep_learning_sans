Feedforward Neural Network Implementation

Overview

This project implements a basic Feedforward Neural Network (FNN) from scratch using Python and libraries such as NumPy and TensorFlow. The purpose of this project is to understand the foundational concepts of neural networks, including forward propagation, activation functions, backpropagation, and model optimization.

Features

Fully connected layers
Configurable number of hidden layers and neurons
Different activation functions (ReLU, Sigmoid, Tanh, etc.)
Loss functions (Mean Squared Error, Cross-Entropy)
Gradient Descent optimization (with learning rate and epochs configuration)
Accuracy and Loss tracking per epoch
Getting Started

Prerequisites
Make sure you have Python 3.x installed. The following Python libraries are also required:

numpy for mathematical operations.
matplotlib for plotting loss and accuracy graphs (optional).
tensorflow or pytorch for simplified neural network operations.
You can install these libraries via pip:

bash
Copy code
pip install numpy matplotlib tensorflow
Running the Code
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/feedforward-neural-network.git
cd feedforward-neural-network
Open the script feedforward_nn.py.
Adjust the configuration of the neural network, such as number of layers, neurons, activation functions, etc.
Run the script:
bash
Copy code
python feedforward_nn.py
The training process will begin, and you will see the accuracy and loss progress for each epoch printed in the console. You may also see a graph of accuracy and loss per epoch if the matplotlib library is installed.
Code Overview

The core components of the implementation are:

Initialization: Define the architecture of the neural network with input, hidden, and output layers.
Forward Propagation: Pass input through each layer using matrix multiplications and activation functions.
Backpropagation: Calculate the gradients of the loss with respect to each parameter and update the weights.
Optimization: Use gradient descent to update weights in the direction that reduces the loss.
Example Usage

Here is an example of configuring and training a simple feedforward neural network for classification on a dataset like MNIST:

python
Copy code
import numpy as np
from feedforward_nn import FeedforwardNeuralNetwork

# Initialize the network with 2 hidden layers
nn = FeedforwardNeuralNetwork(input_size=784, hidden_layers=[128, 64], output_size=10)

# Train the network
nn.train(X_train, y_train, epochs=100, learning_rate=0.01)

# Test the network
accuracy = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}%")
Results

The network's accuracy and loss should improve over time, and it should eventually converge to an optimal solution depending on the dataset and hyperparameters chosen.

Future Improvements

Implementing other optimizers like Adam, RMSProp.
Support for batch normalization and dropout layers.
More advanced activation functions and layer configurations.
License

This project is licensed under the MIT License - see the LICENSE file for details.