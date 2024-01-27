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


class Activation_Softmax: # Activation mostly used for output layer
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        sample_len = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample_len), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        negative_log = -np.log(correct_confidences)

        return negative_log



X, y = spiral_data(samples = 100, classes = 3)


layer_dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

layer_dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

layer_dense1.forward(X)
activation1.forward(layer_dense1.output)

layer_dense2.forward(activation1.output)
activation2.forward(layer_dense2.output)

print(activation2.output[:5])


loss_function = Loss_Categorical_Cross_Entropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss: ", loss)
