import numpy as np

def sigmoid(z):

    sigmoid = 1/(1 + np.exp(-z))
    return sigmoid


def reLu(z):
    if(z > 0):
        relu = z
    else:
        relu = 0

    return relu
    

def softmax(z):
    #z(Bilder, Klassen)
    summ = np.sum(np.exp(z), axis=1, keepdims=True)
    softmax = np.exp(z) / summ

    return softmax


