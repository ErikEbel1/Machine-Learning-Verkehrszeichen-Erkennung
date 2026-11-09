import numpy as np
import helperfunctions as hf

##DATA HANDLING-----------------------------------------------------------------
def forward_propagation(X, parameters):

    A = X
    cache = {"A0": X}

    number_of_layers = int(len(parameters) / 2)

    for i in range(1, number_of_layers + 1):

        W = parameters[f"W{i}"]
        b = parameters[f"b{i}"]

        Z = np.dot(A, W.T) + b

        if i == number_of_layers:
            A = hf.softmax(Z)
        else:
            A = hf.reLu(Z)

        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A

    return A, cache


def compute_cost(A, Y):
    #A(Bilder, probabilitys)  bsp.[20,3,4,36,20,9,...,n]
    #Y Tatsächlichen Werte   
    #Sparse Categorical Cross-Entropy   

    n = A.shape[0]

    actual_props = A[np.arange(n), Y]
    cost = np.mean(- np.sum( np.log(actual_props)))

    return cost
    


##RESTULTS HANDLING

def comp_class_prediction(A):
    #A(Bild, porbabilty)
    prediction = np.argmax(A, axis=1, keepdims= True)


