import sys
import numpy as np
import matplotlib

# Batches are portions of the dataset instead of the entire dataset to perform gradual updates during training
# Using all the data at once could cause over-fitting for a model


inputs = [[1, 3, 4, 2.7],
          [2, 8.6, -0.3, 3],
          [7.4, 2.1, -4.1, 1.2]]

# No need to change weights because no nodes are added

weights = [[2, 7, 9, -1.5],
           [0.5, -0.9, 9, -1.5],
           [1, 0.23, 2, 3.4]]

biases = [3, 1, 0.8]

weights = np.array(weights).T # Transpose (3, 4) --> (4, 3)

output = np.dot(inputs, weights) + biases

print(output)