import numpy as np 

class CategoricalLoss:
    def __init__(self):
        self.name = "categorical"
    
    # returns the loss for the batch of outputs
    def forward(self,y_pred,y_true,one_hot=True):
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if one_hot:
            samples_losses = -np.sum(y_true * np.log(y_pred_clipped), axis = 1)
            
            return np.mean(np.reshape(samples_losses,(-1)))

        else:
            samples_losses = y_pred_clipped[range(len(y_pred_clipped)), y_true]
            return np.mean(-np.log(samples_losses))
        
class MeanSquared:
    def __init__(self):
        self.name = "meanSquared"

    def forward(self,y_pred,y_true):
        return (1 / y_true.size) * np.sum((y_pred - y_true) ** 2)
    

