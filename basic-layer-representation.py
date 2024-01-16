import sys
import numpy as np
import matplotlib


# Represents a layer where there are 4 inputs to a 3 neuron output layer
# For each input[i], weights[i] in all 3 sets associate with it
# Each node has its own unique weight set (equal to the length of inputs)

inputs = [1, 3, 4, 2.7] # Inputs are the same for each node the weights change the values
# Weights for each node
weights1 = [2, 7, 9, -1.5] # Node 1
weights2 = [0.5, -0.9, 9, -1.5] # Node 2
weights3 = [1, 0.23, 2, 3.4] # Node 3

bias1 = 3
bias2 = 1
bias3 = 0.8



output1 = [(inputs[0] * weights1[0]) + (inputs[1] * weights1[1]) + (inputs[2] * weights1[2]) + (inputs[3] * weights1[3]) + bias1]
output2 = [(inputs[0] * weights2[0]) + (inputs[1] * weights2[1]) + (inputs[2] * weights2[2]) + (inputs[3] * weights2[3]) + bias2]
output3 = [(inputs[0] * weights3[0]) + (inputs[1] * weights3[1]) + (inputs[2] * weights3[2]) + (inputs[3] * weights3[3]) + bias1]

print(output1, output2, output3)

