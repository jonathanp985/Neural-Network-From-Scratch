import sys
import numpy as np
import matplotlib

# Used for multiplying differing array shapes together

inputs = [1, 3, 4, 2.7]

weights = [[2, 7, 9, -1.5],
           [0.5, -0.9, 9, -1.5],
           [1, 0.23, 2, 3.4]]

biases = [3, 1, 0.8]

# Putting weights first because of shape compatibility and desire of output shape
output = np.dot(weights, inputs) + biases

print(output)