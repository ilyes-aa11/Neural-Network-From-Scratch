import numpy as np

# "inputs" can be a batch of inputs or a sample
# "y" must be a sample (row vector) of post activation values for a single layer
# calculate_derivative returns the Jacobian matrix of the activation function

class ReLU:
    def __init__(self):
        self.name = "relu"

    def calculate(self,inputs):
        return np.maximum(0,inputs)
    
    def calculate_derivative(self,y):
        return np.diag(np.where(y >= 0,1,0))
        
class Softmax:
    def __init__(self):
        self.name = "softmax"

    def calculate(self, inputs: np.ndarray):
        inputs = np.asarray(inputs, dtype=np.float64)
        # shifting to avoid numerical overflow for large input values
        if inputs.ndim == 2:
            shifted = inputs - np.max(inputs, axis=1, keepdims=True)
            exps = np.exp(shifted)
            sums = np.sum(exps, axis=1, keepdims=True)
            return exps / sums
        elif inputs.ndim == 1:
            shifted = inputs - np.max(inputs)
            exps = np.exp(shifted)
            return exps / np.sum(exps)
        
    def calculate_derivative(self,y):
        n = len(y)
        y_tiled = np.repeat(y , n).reshape((n,n))
        return y_tiled * (np.identity(n) - np.transpose(y_tiled))

class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"

    def calculate(self,inputs):
        return 1 / (1 + np.exp(-inputs));

    def calculate_derivative(self,y):
        return np.diag(y * (1 - y))
