import csv
import numpy as np
from PIL import Image



def load_dataset():
    #Funktion zum Laden des Datensatzes und der lables

    #csv datei einlesen
    # CSV-Dateien einlesen

    with open("data/Train.csv", "r", encoding="utf-8") as file:
        csv_train_set_tabular = list(csv.DictReader(file))

    with open("data/Test.csv", "r", encoding="utf-8") as file:
        csv_test_set_tabular = list(csv.DictReader(file))
    
    #Ergebnisse
    train_set_values = np.array([int(row["ClassId"]) for row in csv_train_set_tabular])
    test_set_values = np.array([int(row["ClassId"]) for row in csv_test_set_tabular])

    #Daten als RGB Array abspeichern
    test_data = np.array([
        np.asarray(
            Image.open("data/" + row["Path"])
            .convert("RGB")
            .resize((48, 48))
        )
        for row in csv_test_set_tabular
    ])


    train_data = np.array([
        np.asarray(
            Image.open("data/" + row["Path"])
            .convert("RGB")
            .resize((48, 48))
        )
        for row in csv_train_set_tabular
    ])

    return test_data, test_set_values, train_data, train_set_values    





