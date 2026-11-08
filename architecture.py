import numpy as np

LEARNING_RATE = 0.01
CLASSES_SIZE = 43
HIDDEN_LAYER_SIZE = 100


def load_hyperparameters(data_set_flatten):
    X = data_set_flatten


    hyper_parameters = { 
        "input_layer_size" : X.shape[1],
        "hidden_layer_size" : HIDDEN_LAYER_SIZE,
        "output_layer_size" : CLASSES_SIZE,
        "learning_rate" : LEARNING_RATE
    }

    return hyper_parameters