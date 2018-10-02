import shutil
import os
import pandas as pd
import glob

path = "E:/OneDrive/Mestrado/Final assignment/datasets/nexet/"

"""
Separa as imagens do dataset nexet in pastas (uma pasta por classe)
"""

def get_next_labels(files):
    """
    Retorna os lables para o dataset next
    """
    df = load_csv_file("train.csv")
    y_true = []

    for file in files:
        name = file.replace("\\", "/")
        name = name.split("/")[-1]

        filter = df[df["image_filename"].str.contains(name)]
        rotulo = filter.iloc[0][1]
        if rotulo == "Day":
            y_true.append(0)
        elif rotulo == "Night": 
            y_true.append(1)
        else:
            y_true.append(2)
    
    return y_true
    
     
# values = util.load_csv_file("output/values.csv").values
# copy_classified_image(values)


def load_csv_file(file_name):
    return pd.read_csv(path + file_name)


def get_all_files_nexet():
    """
    Retorna uma lista com todos os arquivos jpg no path indicado
    """
    return glob.glob("E:/OneDrive/Mestrado/Final assignment/datasets/nexet/*.jpg")


def label_dataset_manually():
    df = load_csv_file("train.csv")

    # shutil.rmtree(path + "Dia", ignore_errors=True)
    # shutil.rmtree(path + "Noite", ignore_errors=True)
    # shutil.rmtree(path + "Transicao", ignore_errors=True)

    # os.mkdir(path + "Dia")
    # os.mkdir(path + "Noite")
    # os.mkdir(path + "Transicao")

    for index, row in df.iterrows():
        print(index)

        file_name = row[0]
        rotulo = row[1]

        try:
            if rotulo == "Day":
                shutil.copyfile(path + row[0], path + "Dia/" + file_name)
            elif rotulo == "Night":
                shutil.copyfile(path + row[0], path + "Noite/" + file_name)
            else:
                shutil.copyfile(path + row[0], path + "Transicao/" + file_name)

        except Exception as ex:
            pass
