import util
import numpy as np
import multiprocessing
import process_dataset_transient as pdt

from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.xception import Xception, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from pandas import DataFrame as df
from joblib import Parallel, delayed                            


def get_CNN_features(files_names=None):
    """
    Extrai features com CNNs pre-treinadas e os retorna
    """
    model = ResNet50(include_top=False, weights='imagenet')
    # model = InceptionV3(include_top=False, weights='imagenet')
    # model = Xception(include_top=False, weights='imagenet')

    if files_names is None:
        files1 = util.get_all_files_names("output/Dia/")
        files2 = util.get_all_files_names("output/Noite/")
        files3 = util.get_all_files_names("output/Trasnsicao/")

        files_names = files1 + files2 + files3
        
    tensors = util.paths_to_tensor(files_names)
    tensors = preprocess_input(tensors)
    pred = model.predict(tensors).reshape(len(files_names), 2048)

    return pred


'''
if __name__ == '__main__':
    pred = get_CNN_features()
    df(pred).to_csv(util.path + "output/x_cnn_xcep.csv", mode='a', header=True, index=False)
'''