import layer
import activation
import loss
import neuralnet
import numpy as np
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

X,y = load_digits(return_X_y=True)
X = X / 16.0


# converting y to one-hot encoding
y_onehot = np.zeros((y.shape[0],10))
for i in range(len(y)):
    y_onehot[i][y[i]] = 1

# splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y_onehot,test_size=0.2)

# constructing the nueral network
layer1 = layer.DenseLayer(X.shape[1],32,activation.ReLU())
layer2 = layer.DenseLayer(32,16,activation.ReLU())
layer3 = layer.DenseLayer(16,10,activation.Softmax())

net = neuralnet.NeuralNet(layer1,layer2,layer3,loss=loss.CategoricalLoss())


# optimization
net.fit(X_train,y_train)
# testing
y_pred = net.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1) 

print("accuracy: " ,accuracy_score(y_true=y_test,y_pred=y_pred)) # accuracy between 97% to 90%
