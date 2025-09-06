import layer
import activation
import loss
import optimizer
import numpy as np
from sklearn.datasets import load_digits # dataset for hand written digits a lighter version of the mnist dataset
from sklearn.model_selection import train_test_split # used to split the dataset into training data & testing data
from sklearn.metrics import accuracy_score # evaluates the final score of the network i.e the percentage of how well it performed

digits = load_digits()
X , y = digits.data, digits.target
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


lossFunction = loss.CategoricalLoss()
# optimization
optimizer.stochasticGradientDescent([layer1,layer2,layer3],lossFunction,X_train,y_train)

# testing
layer1.forward(X_test)
layer1.activate()
layer2.forward(layer1.activated)
layer2.activate()
layer3.forward(layer2.activated)
layer3.activate()

print("Testing loss: ",lossFunction.forward(layer3.activated, y_test))

y_pred = np.argmax(layer3.activated, axis=1)
y_test = np.argmax(y_test, axis=1) 

print("accuracy: " ,accuracy_score(y_true=y_test,y_pred=y_pred)) # accuracy between 97% to 90%
