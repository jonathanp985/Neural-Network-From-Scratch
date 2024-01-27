import numpy as np
import nnfs
from nnfs.datasets import spiral_data

"""
An Activation Function decides whether a neuron should be activated or not. 
This means that it will decide whether the neuron's input to the 
network is important or not in the process of prediction using simpler mathematical operations.
"""

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Distribution is < 1
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # Function of either 0 or X



inputs = [[1, 3, 4, 2.7],
          [2, 8.6, -0.3, 3],
          [7.4, 2.1, -4.1, 1.2]]

X, y =  spiral_data(100, 3)
layer1 = Layer_Dense(2, 10)

activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)