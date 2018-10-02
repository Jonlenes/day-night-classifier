import util
import shutil
import os

sub_folder = "imageAlignedLD/"
path_class = util.path + "output/"

def get_unclassified_imagens():
    """
    Retorna todas as imagens do transient que ainda nao foram adicionadas a nosso dataset
    """
    df = util.load_tsv_file("/annotations/annotations.tsv")
    paths = []

    for _, row in df.iterrows():
        file_name = row[0]

        file_name_new = file_name.split("/")
        file_name_new = file_name_new[0] + "_" + file_name_new[1]
        
        if (not util.os.path.exists(path_class + "Trasnsicao/" + file_name_new) and
           not util.os.path.exists(path_class + "Noite/" + file_name_new) and
           not util.os.path.exists(path_class + "Dia/" + file_name_new)):

            paths.append(util.path + sub_folder + file_name)
    
    return paths


def copy_classified_image(values):
    """
    Dado um conjunto de imagens ja classificadas, copia as para uma de avaliacao manual 
    """
    shutil.rmtree(path_class + "0", ignore_errors=True)
    shutil.rmtree(path_class + "1", ignore_errors=True)
    shutil.rmtree(path_class + "2", ignore_errors=True)
 
    util.os.mkdir(path_class + "0")
    util.os.mkdir(path_class + "1")
    util.os.mkdir(path_class + "2")
 
    for value in values:
        if int(value[1]) != -1:
            file_name = value[0]
    
            file_name = file_name.split("/")
            file_name_new = file_name[-2] + "_" + file_name[-1]
            
            shutil.copyfile(value[0], path_class + value[1] + "/" + file_name_new)
     
# values = util.load_csv_file("output/values.csv").values
# copy_classified_image(values)


def label_dataset_manually():
    """
    Classifica o transient baseados somente nos seus atributos
    """
    df = util.load_tsv_file("/annotations/annotations.tsv")
    
    shutil.rmtree(path_class + "Dia", ignore_errors=True)
    shutil.rmtree(path_class + "Noite", ignore_errors=True)
    shutil.rmtree(path_class + "Diferente", ignore_errors=True)
    
    util.os.mkdir(path_class + "Dia")
    util.os.mkdir(path_class + "Noite")
    util.os.mkdir(path_class + "Diferente")
    
    for index, row in df.iterrows():
        print(index)
        file_name, daylight, night, sunrisesunset, dawndusk, sunny = row[0], row[2], row[3], row[4], row[5], row[6]

        perct_daylight = float(daylight.split(",")[0])
        perct_night = float(night.split(",")[0])
        perct_sunrisesunset = float(sunrisesunset.split(",")[0])
        perct_dawndusk = float(dawndusk.split(",")[0])
        perct_sunny = float(sunny.split(",")[0]) 

        file_name_new = file_name.split("/")
        file_name_new = file_name_new[0] + "_" + file_name_new[1]
        
        if perct_sunrisesunset > .8 or perct_dawndusk > .8:

            shutil.copyfile(util.path + sub_folder + file_name, path_class + "Diferente/" + file_name_new)

        elif perct_night > .8:

            shutil.copyfile(util.path + sub_folder + file_name, path_class + "Noite/" + file_name_new)

        elif perct_daylight > .8 or perct_sunny > .8:
            
            shutil.copyfile(util.path + sub_folder + file_name, path_class + "Dia/" + file_name_new)
        
    
