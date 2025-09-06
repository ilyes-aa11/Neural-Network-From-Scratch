import numpy as np 

def ActivationDerivative(Layer,sample_index):
    match Layer.activation.name:
        case "sigmoid":
            return Layer.activated[sample_index] * (1 - Layer.activated[sample_index])
        
        case "relu":
            return np.minimum(np.ceil(np.maximum(Layer.output[sample_index],0)),1) 
        
        case "softmax":
            return Layer.activated[sample_index] * (1 - Layer.activated[sample_index])
        

def LossDerivative(OutputLayer,sample_index,Y_true,LossFunction):
    match LossFunction.name:
        case "meanSquared":
            return (2/Y_true[sample_index].size) * (OutputLayer.activated[sample_index] - Y_true[sample_index])
        
        case "categorical":
            Y_pred_clipped = np.clip(OutputLayer.activated[sample_index],1e-7,1-1e-7)
            return - Y_true[sample_index] / Y_pred_clipped
        

def Back_propagate(Layers, layer_index, sample_index, x_train, y_true, learning_rate,/, output_layer=False, Loss=None, previous_error=None):
    if layer_index < 0: # base case
        return
    
    if output_layer:
        if Layers[layer_index].activation.name == "softmax" and Loss.name == "categorical":
            curr_error = Layers[layer_index].activated[sample_index] - y_true[sample_index]
        else:
            d_activation = ActivationDerivative(Layers[layer_index],sample_index)
            d_loss = LossDerivative(Layers[layer_index], sample_index, y_true, Loss)
            curr_error = d_activation*d_loss
        Back_propagate(Layers, layer_index-1, sample_index, x_train, y_true, learning_rate, previous_error=curr_error)
        if layer_index > 0:
            deltaW = learning_rate * np.outer(Layers[layer_index - 1].activated[sample_index], curr_error)
        else:
            deltaW = learning_rate * np.outer(x_train[sample_index], curr_error)
        deltaB = learning_rate * curr_error
        
        Layers[layer_index].weights -= deltaW
        Layers[layer_index].bias -= deltaB

    else:
        d_activation = ActivationDerivative(Layers[layer_index],sample_index) 
        curr_error = d_activation * np.dot(Layers[layer_index+1].weights,previous_error) 
        Back_propagate(Layers, layer_index-1, sample_index, x_train, y_true, learning_rate, previous_error=curr_error)
        if layer_index > 0:
            deltaW = learning_rate * np.outer(Layers[layer_index - 1].activated[sample_index], curr_error)
        else:
            deltaW = learning_rate * np.outer(x_train[sample_index], curr_error)
        deltaB = learning_rate * curr_error
        Layers[layer_index].weights -= deltaW
        Layers[layer_index].bias -= deltaB

def ForwardInput(Layers,x_train):
    for i in range(len(Layers)):
        if i == 0:
            Layers[i].forward(x_train)
            Layers[i].activate()
        else:
            Layers[i].forward(Layers[i-1].activated)
            Layers[i].activate()

def stochasticGradientDescent(Layers,Loss,x_train,y_true):
    learning_rate = 0.01
    iterations = 30

    for _ in range(iterations):
        for s in range(x_train.shape[0]): # backpropagate for each sample
            ForwardInput(Layers,x_train)
            Back_propagate(Layers, len(Layers) - 1, s, x_train, y_true, learning_rate, output_layer=True, Loss=Loss)
            
        print(_, Loss.forward(Layers[-1].activated,y_true))

        

        
