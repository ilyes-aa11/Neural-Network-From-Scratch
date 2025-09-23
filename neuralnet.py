import numpy as np

class NeuralNet:
    def __init__(self,*layers,loss):
        self.layers = [layer for layer in layers]
        self.loss = loss # loss function

    def _forward(self,inputs):
        X = inputs
        for layer in self.layers:
            layer.forward(X)
            layer.activate()
            X = layer.activated

    def _backpropagation(self,y_true,sample,learning_rate):
        outputLayer = self.layers[-1]
       
        dL_dZ = self.loss.calculate_derivative(outputLayer.activated[sample],y_true[sample]) 
        dL_dZ = np.reshape(dL_dZ,(1,-1))

        for i in range(len(self.layers)-1,-1,-1):
            dL_dZ = self.layers[i].backward(dL_dZ,sample,learning_rate)

    def fit(self,X_train,y_train):
        epoch = 100
        learning_rate = 0.001
        epsilon = 1e-3

        self._forward(X_train)
        old_loss = self.loss.calculate(self.layers[-1].activated,y_train)
        for i in range(epoch):
            print(i, old_loss)
            for j in range(X_train.shape[0]): # for each sample
                self._forward(X_train)
                self._backpropagation(y_train,j,learning_rate)
            
            new_loss = self.loss.calculate(self.layers[-1].activated,y_train)
            if abs(old_loss - new_loss) < epsilon:
                break
            old_loss = new_loss


    def predict(self,X_test):
        self._forward(X_test)
        return self.layers[-1].activated