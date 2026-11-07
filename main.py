from dataset import load_dataset
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


test_data, test_set_values, train_data, train_set_values  = load_dataset() 

index = 10

plt.imshow(test_data[index])
plt.title(f"Klasse: {train_set_values[index]}")
plt.axis("off")
plt.show()
