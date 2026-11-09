from dataset import load_dataset
from architecture import load_hyperparameters
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#Daten Vorbereitung-----------------------
test_data_orig, test_set_values, train_data_orig, train_set_values  = load_dataset() 

#reshaping
test_data_flatten = test_data_orig.reshape(test_data_orig.shape[0], -1) #(shape==(Bilder, 6912))

#Broadcast der Pixelwerte
test_data_flatten = test_data_flatten / 255

#Initialisierungen-------------------------

hyper_parameters = load_hyperparameters(test_data_flatten)

n_x = hyper_parameters["input_layer_size"]
n_h1 = hyper_parameters["hidden_layer_size"]
n_y = hyper_parameters["output_layer_size"]
#     zum hinzufügen von weiteren hidden layers

parameters = {
    "W1": np.random.randn(n_h1, n_x) * 0.01,
    "b1": np.zeros((1,n_h1)),
    "W2": np.random.randn(n_y, n_h1 ) * 0.01,
    "b2": np.zeros((1, n_y))
}


print("parameters: ", parameters)
print(test_data_flatten.shape)  

# plt.imshow(test_data[index])
# plt.title(f"Klasse: {train_set_values[index]}")
# plt.axis("off")
# plt.show()
