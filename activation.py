import numpy as np

class ReLU:
    def __init__(self):
        self.name = "relu"

    def forward(self,inputs):
        return np.maximum(0,inputs)
        
class Softmax:
    def __init__(self):
        self.name = "softmax"

    def forward(self, inputs: np.ndarray):
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

class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"

    def forward(self,inputs):
        return 1 / (1 + np.exp(-inputs));



