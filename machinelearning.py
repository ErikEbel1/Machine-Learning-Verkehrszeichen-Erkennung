import numpy as np
import helperfunctions as hf


def forward_propagation(X, parameters):

    
    #Extrahierung aus parameters obj ( bisther 1 h_layer)
    W1 = parameters["W1"]   #(h_n, h_x)
    b1 = parameters["b1"]   #(1, h_n)

    W2 = parameters["W2"]   #(y_n, h_n)
    b2 = parameters["b2"]   #(1, y_n)

    #X ist der Datensatz shape == (Bilder = n , Features = 6912 = h_x)
    Z1 = np.dot(W1, X.T) + b1  #(n, h_n)
    #Activation Funktion 1 (ReLu?)
    A1 =                #(n, h_n)

    Z2 = np.dot(W2, A1.T) + b2
    A2 = hf.sigmoid(Z2)


    return A2