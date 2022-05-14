
from numpy import exp


def relu(x):
    if x > 0:
        return x
    return 0

def relu_derivative(x):
    if x > 0 :
        return 1
    return 0

def sigmoid(x):
    f = 1/(1+exp(-x))
    return f

def sigmoid_derivative(x):
    df = sigmoid(x)*(1-sigmoid(x))
    return df

def liniar(x):
    return x

def liniar_derivative(x):
    return 1