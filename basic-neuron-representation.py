import sys
import numpy as np
import matplotlib

# code below represents a single neuron in a full network
# 4 neurons feeding into the neuron

inputs = [1, 3, 4, 2.7] # outputs from previous neurons
weights = [2, 7, 9, -1.5] # Every input has a unique weight associated with it
bias = 3 # every unique neuron has a unique bias

output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + (inputs[0] * weights[0]) + bias # how output is calculated
print(output)

