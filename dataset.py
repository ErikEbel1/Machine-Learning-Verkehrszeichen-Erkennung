import csv
import numpy as np



def load_dataset():
    #Funktion zum Laden des Datensatzes und der lables
    # 32 x 32 Pixel 

    #csv datei einlesen
        # CSV-Dateien einlesen
    with open("data/Train.csv", "r", encoding="utf-8") as file:
        csv_train_set_tabular = list(csv.DictReader(file))

    with open("data/Test.csv", "r", encoding="utf-8") as file:
        csv_test_set_tabular = list(csv.DictReader(file))
    
    #Ergebnisse
    train_set_values = np.array([int(row["ClassId"]) for row in csv_train_set_tabular])
    test_set_values = np.array([int(row["ClassId"]) for row in csv_test_set_tabular])

    #Daten-location
    test_data = np.array([row["Path"] for row in csv_test_set_tabular])
    train_data = np.array([row["Path"] for row in csv_train_set_tabular])

    return test_data, test_set_values, train_data, train_set_values    





