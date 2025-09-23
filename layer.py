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
        self.inputs = inputs
        self.output =  np.dot(inputs, self.weights) + self.bias

    def activate(self):
        self.activated = self.activation.calculate(self.output)

    def backward(self,dL_dZ,sample,learning_rate):
        # takes dL/dZ returns dL/dX
        # Z denotes post activation values for a layer 
        # Y denotes pre activation values for a layer
        dL_dY = np.dot(dL_dZ,self.activation.calculate_derivative(self.activated[sample]))
        dY_dX = self.weights.T
        dL_dX = np.dot(dL_dY, dY_dX)
        dL_dW = np.dot(np.reshape(dL_dY,(-1,1)), np.reshape(self.inputs[sample],(1,-1))) 
        # dL_dB = dL_dY

        # gradient descent update for weights & bias
        self.weights = self.weights - learning_rate * np.transpose(dL_dW)
        self.bias = self.bias - learning_rate * dL_dY

        return dL_dX
