import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Distribution is < 1
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # Function of either 0 or X

"""
Takes the exponential values to "e" 
Then divides those values to the sum of that input to create a normalized data set
To prevent overflow each input is subtracted by the max of that input then the exponential is taken
"""

class Activation_Softmax: # Activation mostly used for output layer
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities


X, y = spiral_data(samples = 100, classes = 3)

layer_dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

layer_dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

layer_dense1.forward(X)
activation1.forward(layer_dense1.output)

layer_dense2.forward(activation1.output)
activation2.forward(layer_dense2.output)

print(activation2.output)

