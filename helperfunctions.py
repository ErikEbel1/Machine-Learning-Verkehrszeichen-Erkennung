import numpy as np

def sigmoid(z):

    sigmoid = 1/(1 + np.exp(-z))
    return sigmoid



