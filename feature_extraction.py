import util
import process_dataset_transient as pdt

from pandas import DataFrame as df
from time import time
from feature_extraction_CNN import get_CNN_features
from skimage.feature import greycoprops, greycomatrix, local_binary_pattern


HISTOGRAM_FEATURES = True
COMATRIX_FEATURES = True
LBP_FEATURES = False
CNN_FEATURES = True


def feature_extract(image, i, size, param):
    """
    Dada uma image, extrai hist, co-matrix e LBP features de acordo com param
    """
    util.printProgressBar(i, size)
    features = []

    for ch in range(3):
        ch_img = image[:, :, ch]
        ch_img = (ch_img / 8)
        ch_img = ch_img.astype(util.np.int8)
        
        if param[0]:
            hist = util.cv2.calcHist([image], [ch], None, [256], [0,256])
            features.extend( hist.flatten() )

        if param[1]:
            co_matrix = greycomatrix(ch_img, [5], [0], 32, symmetric=True, normed=True)
            features.extend( co_matrix.flatten() )

        if param[2]:
            features.extend( get_lbp_features(image[:, :, ch], 12, 4) )
    
    return util.np.array(features).ravel().flatten()


def get_lbp_features(image, num_points, radius, eps=1e-7):
    """
    Computa o lbp features
    """
    lbp = local_binary_pattern(image, num_points, radius, method="uniform")
    (hist, _) = util.np.histogram(lbp.ravel(), bins=util.np.arange(0, num_points + 3), range=(0, num_points + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def get_statistics(a):
    """
    Calcula as estastiticas sobre uma matriz
    """
    a = a.flatten()
    return [util.np.mean(a), util.np.var(a), util.skew(a), util.kurtosis(a), util.np.std(a)]  


def feature_extract_parallel(imgs, histogram=False, comatrix=False, lbp=False):
    """
    Executa a extracao de features em paralelo usado num_cores threads
    """
    t = time()
    files_count = len(imgs)
    param = [histogram, comatrix, lbp]

    print("Extração de features: %d cores." % util.num_cores)

    list_of_features = util.Parallel(n_jobs=util.num_cores)(
        util.delayed(feature_extract)(imgs[i], i, files_count, param) for i in range(files_count))
    
    print("\nTime: ", time() - t)
    
    return list_of_features 

def extract_and_save_features(unclassifier=False, paths=None, index_csv=0):
    """
    Extrai e salva os features em aqruivos CSV
    """
    imgs = []
    if HISTOGRAM_FEATURES or COMATRIX_FEATURES or LBP_FEATURES:
        if unclassifier:
            imgs = util.load_images_parallel(files=paths)
        else:
            imgs_day = util.load_images_parallel("output/Dia/")
            imgs_night = util.load_images_parallel("output/Noite/")
            imgs_diff = util.load_images_parallel("output/Trasnsicao/")

            imgs = imgs_day + imgs_night + imgs_diff
            y = [0] * len(imgs_day) + [1] * len(imgs_night) + [2] * len(imgs_diff)
            df(y).to_csv(util.path + "output/y.csv", mode='a', header=True, index=False)


        if HISTOGRAM_FEATURES:
            x = feature_extract_parallel(imgs, histogram=True) 
            df(x).to_csv(util.path + "output/x_hist" + str(index_csv) + ".csv", mode='a', header=True, index=False)

        if COMATRIX_FEATURES:
            x = feature_extract_parallel(imgs, comatrix=True)
            df(x).to_csv(util.path + "output/x_co_matrix" + str(index_csv) + ".csv", mode='a', header=True, index=False) 

        if LBP_FEATURES:
            x = feature_extract_parallel(imgs, lbp=True)
            df(x).to_csv(util.path + "output/x_lbp" + str(index_csv) + ".csv", mode='a', header=True, index=False) 
        
        del imgs

    if CNN_FEATURES:
        x = get_CNN_features(paths)
        df(x).to_csv(util.path + "output/x_cnn" + str(index_csv) + ".csv", mode='a', header=True, index=False)


if __name__ == '__main__':
    
    '''
    import random
    paths=util.get_amos_files()
    news = []

    for i in range(10000):
        index = random.randint(10000, 500000)
        news.append(paths[index])
    df(news).to_csv(util.path + "output/news.csv", header=True, index=False)

    extract_and_save_features(unclassifier=True, paths=news, index_csv=2)
    '''
    # extract_and_save_features(unclassifier=True, paths=pdn.get_unclassified_imagens(), index_csv=1)
     
    import process_dataset_nexet as pdn
    pdn.get_next_labels(pdn.get_all_files_nexet()[0:10000])
    extract_and_save_features(unclassifier=True, paths=pdn.get_all_files_nexet()[0:10000], index_csv=5)
    # extract_and_save_features()