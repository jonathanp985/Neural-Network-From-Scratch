import math

import numpy as np

# Catagorical loss entropy (classification loss function)
softmax_outputs = np.array([[0.7, 3, 4,],
                            [2, 8.6, -0.3],
                            [7.4, 2.1, -4.1]])

target_output = [0, 1, 1]

loss = -np.log(softmax_outputs[range(len(softmax_outputs)), target_output])
print(loss)

average_loss = np.mean(loss)
print(average_loss)