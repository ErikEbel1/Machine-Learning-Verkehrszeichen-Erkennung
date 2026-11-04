from dataset import load_dataset
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


test_data, test_set_values, train_data, train_set_values  = load_dataset() 

index = 100

image = Image.open(Path("data") / train_data[index])

plt.imshow(image)
plt.title(f"Klasse: {train_set_values[index]}")
plt.axis("off")
plt.show()
