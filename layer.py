import numpy as np

class DenseLayer:
    def __init__(self,input_size,layer_size,activation_function):
        self.input_size = input_size
        self.layer_size = layer_size
        
        if activation_function.name == "relu":
            self.weights = np.random.randn(self.input_size, self.layer_size) * np.sqrt(2 / self.input_size)
        else:
            self.weights = np.random.randn(self.input_size, self.layer_size) * np.sqrt(1 / self.input_size)

        self.bias = np.zeros(self.layer_size)

        self.activation = activation_function

    def forward(self,inputs: np.ndarray): # inputs must be a 2d matrix each row represents one sample
        self.output =  np.dot(inputs, self.weights) + self.bias

    def activate(self):
        self.activated = self.activation.forward(self.output)
