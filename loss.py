import numpy as np 

class CategoricalLoss:
    def __init__(self):
        self.name = "categorical"
    
    # returns the loss for the batch of outputs
    def calculate(self,y_pred,y_true):
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        samples_losses = -np.sum(y_true * np.log(y_pred_clipped), axis = 1)
        return np.mean(np.reshape(samples_losses,(-1)))
        
    def calculate_derivative(self,y_pred,y_true):
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        return -y_true / y_pred_clipped
        
class MeanSquared:
    def __init__(self):
        self.name = "meanSquared"

    def calculate(self,y_pred,y_true):
        return (1 / y_true.size) * np.sum((y_pred - y_true) ** 2)
    
    def calculate_derivative(self,y_pred,y_true) :
        return (2 / y_true.size) * (y_pred - y_true)
