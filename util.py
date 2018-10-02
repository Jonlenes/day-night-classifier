import os
import glob
import numpy as np
from joblib import Parallel, delayed                            
import multiprocessing
import cv2
import pandas as pd
from time import time
from scipy.stats import kurtosis, skew
from keras.preprocessing.image import load_img, img_to_array

path = "C:/Users/Kchristtyne/Desktop/datasetraf/Transient/"
path_AMOS = "C:/Users/Kchristtyne/Desktop/datasetraf/AMOS/raw_data/"
num_cores = multiprocessing.cpu_count()


def load_tsv_file(file_name):
    """
    Carrega um arquivo tsv com nome file_name que esteja no diretorio path
    """
    return pd.read_csv(path + file_name, sep='\t', names=range(0, 42))


def load_csv_file(file_name, local_path=path):
    """
    Carrega um arquivo CSV com nome file_name que esteja no diretorio local_path
    """
    return pd.read_csv(local_path + file_name)


def get_all_files_names(sub_folder="", path_folder=path):
    """
    Retorna uma lista com todos os arquivos jpg no path indicado
    """
    return glob.glob(path_folder + sub_folder + "*.jpg")


def load_images_parallel(folder="", files=None):
    """
    Carrega em paralelo e retorna todas as imagens na lista de files
    """
    t = time()
    if files is None:
        files = get_all_files_names(folder)
    files_count = len(files)

    print("Carregando imagens: %d cores." % num_cores)

    list_of_imgs = Parallel(n_jobs=num_cores)(delayed(_load_image)(files[i], i , files_count) for i in range(files_count))
    print("\nTime: ", time() - t)
    
    return list_of_imgs 


def _load_image(file, i, size):
    """
    Carrega e realiza o resize de uma imagem
    """
    printProgressBar(i, size)
    img = cv2.imread(file)
    return cv2.resize(img, (512, 512))


def get_amos_files():
    """
    Retorna uma lista com todos os arquivos jpg no path indicado e seus sub diretorios
    """
    return glob.glob(path_AMOS + "/**/*.jpg", recursive=True)


def path_to_tensor(img_path, i):
    """
    Carrega uma imagem e converte para tensor
    """
    print("Processando %d" % i)
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """
    Carrega um conjunto de imagens em paralelo e converte para tensores
    """
    print("Processando com %d cores..." % num_cores)

    list_of_tensors = Parallel(n_jobs=num_cores)(delayed(path_to_tensor)(img_paths[i], i) for i in range(len(img_paths)))
    return np.vstack(list_of_tensors)


def printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 0, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:d}").format(round(100 * ((iteration) / float(total))))
    filledLength = int(length * (iteration) // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
