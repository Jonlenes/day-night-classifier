import util
import shutil
import os

path_class = util.path + "output/"

def copy_classified_image_AMOS(values):
    """
    Dado um conjunto de imagens ja classificadas, copia as para uma de avaliacao manual ()
    """
    shutil.rmtree(path_class + "00", ignore_errors=True)
    shutil.rmtree(path_class + "01", ignore_errors=True)
    shutil.rmtree(path_class + "02", ignore_errors=True)

    util.os.mkdir(path_class + "00")
    util.os.mkdir(path_class + "01")
    util.os.mkdir(path_class + "02")

    count = 0

    for value in values:
        if int(value[1]) != -1:
            value[0] = value[0].replace("\\", "/")
            file_name = value[0]
            
            file_name = file_name.split("/")
            file_name_new = str(count) + "_" + file_name[-1]
            count += 1
            
            shutil.copyfile(value[0], path_class + "0" + str(value[1]) + "/" + file_name_new)
