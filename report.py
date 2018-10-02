from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from pandas import DataFrame as df
import pandas as pd
import glob
import os
import random

""""
files = glob.glob("E:/OneDrive/Mestrado/Final assignment/datasets/Own dataset/" + "/**/*.jpg", recursive=True)
count = 0 
while len(files) > 0:
    index = random.randint(0, len(files) - 1)
    name = files[index]
    files.remove(files[index])
    name = name.replace("\\", "/")
    os.rename(name, "E:/OneDrive/Mestrado/Final assignment/datasets/Own dataset/" + name.split("/")[-2] + "/Own_dataset_" + str(count) + ".jpg" )
    count += 1
"""

path = "C:/Users/Kchristtyne/Desktop/datasetraf/Transient/output/"

def load_csv_file(file_name):
    """
    Carrega um arquivo CSV com nome file_name que esteja no diretorio local_path
    """
    return pd.read_csv(path + file_name)

# 3 e 5
y_val = load_csv_file("y_true.csv").values
pred = load_csv_file("y_pred_3.csv").values

print( classification_report(y_val, pred, target_names=["Dia", "Noite", "Transição"]) )
print( confusion_matrix(y_val, pred) )
