import util
import process_dataset_transient as pdt
import process_dataset_nexet as pdn

from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from keras.models import load_model 
from vgg16_places_365 import preprocess_input as pi
from pandas import DataFrame as df

TEST_ALL_CLASSIFIERS = 0
TEST_BIG_CLASSIFIER = 1


def get_dataset_train(index_csv=0, load_label=True):
    """
    Carrega o dataset que já deve ter sido extraido e salvo em um arquivo csv
    """
    x_hist = util.load_csv_file("output/x_hist" + str(index_csv) + ".csv").values
    x_co_matrix = util.load_csv_file("output/x_co_matrix" + str(index_csv) + ".csv").values
    x_lbp = [] # util.load_csv_file("output/x_lbp" + str(index_csv) + ".csv").values
    x_cnn = util.load_csv_file("output/x_cnn" + str(index_csv) + ".csv").values

    y = []
    if load_label:
        y = util.load_csv_file("output/y.csv").values.ravel()

    return [x_hist, x_co_matrix, x_lbp, x_cnn], y


def predict_score(clf, x=None, y=None, pred=None):
    """
    Realiza a predição sobre x em uma classificador treinado, imprime e retorna o f1-score
    """
    if pred is None:
        pred = clf.predict(x)
    f1 = f1_score(y, pred, average='weighted')
    print("accuracy:", accuracy_score(y, pred), "f1:", f1 )
    return f1


def split_dataset_by_index(indexs, x):
    """
    Dado um dataset x e os indexs selecionando em cross-validation, retorna os exemplos que 
    devem ser utilizados
    """
    x_result = []
    for i in indexs:
        x_result.append(x[i])
    return util.np.array(x_result)


def mounth_meta_x(clf, meta_x, x, pred=None):
    """
    Dado um classificador treinado e conjunto de exemplos (x), faz-se a predição dos exemplos 
    e adiciona em um vetor de meta no eixo 1
    """
    if pred is None:
        pred = clf.predict(x)
    pred = pred.reshape(len(pred), 1)
    if len(meta_x) == 0:
        meta_x = pred
    else:
        meta_x = util.np.append(meta_x, pred, axis=1)
    return meta_x


def concatenate_features_by_index(xs, xi):
    """
    Dados diversos conjuntos de features extraidos das mais diversas formas para os mesmos exemplos,
    retorno um conjunto formado por um subconjunto de conjuntos de features (baseados em xi)
    """
    x = xs[xi[0]]
    for i in range(1, len(xi)):
        x = util.np.append(x, xs[xi[i]], axis=1)
    return x


def concatenate_matrix(x1, x2):
    """
    Concatena duas matrizes no eixo 1
    """
    if len(x1) == 0:
        return x2
    return  util.np.append(x1, x2, axis=1)


def execute_classification(x_train, y_train, x_val, y_val):
    """
    Treina um conjunto de classificadores sobre o x_train e y_train e valida sobre x_val, y_val 
    """
    names = ["Linear SVM", "Decision Tree", "Random Forest", "Neural Net", "Logistic Regression"]

    classifiers = [
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=-1, n_estimators=100),
        MLPClassifier(max_iter=500),
        LogisticRegression()
    ]

    count = 1
    print("Starting the training...")
    for name, clf in zip(names, classifiers):     
        t = time()
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        df(pred).to_csv(util.path + "output/y_pred_" + str(count) + ".csv", header=True, index=False)
        print(name + ":", "accuracy:", accuracy_score(y_val, pred), "Time:", time() - t, "f1:", f1_score(y_val, pred, average='weighted'))
        count += 1

    df(y_val).to_csv(util.path + "output/y_true.csv", header=True, index=False)

def get_deep_model():
    return load_model("saved_models/weights.best.vgg.hdf5")


def load_dataset_tensors(nexet=False):
    """
    Carrega as imagens e converte para tenors para treinar a deep
    """
    if nexet:
        files_names = pdn.get_all_files_nexet()[0:10000]
    else:
        files1 = util.get_all_files_names("output/Dia/")
        files2 = util.get_all_files_names("output/Noite/")
        files3 = util.get_all_files_names("output/Trasnsicao/")

        files_names = files1 + files2 + files3
    
    x = util.paths_to_tensor(files_names)
    x = pi(x)

    return x
 
 
if __name__ == '__main__':

    # MODEL_ACTION
    #   TEST_ALL_CLASSIFIERS - Testa todos os classificadores sobre os features selecionados
    #   TEST_BIG_CLASSIFIER  - Monta o big classificador apresentado e explicado no "artigo"
    #                           gerado desse trabalho
    #       IMAGE_ROTULATION - Rotulas novas imagens com o big classificar, sendo a nova imagem
    #                           adiciona ao dataset somente se todos os classificadores gerarem 
    #                           a mesma classe
    #           AMOS - Rotula dados do dataset AMOS
    MODEL_ACTION = TEST_BIG_CLASSIFIER
    IMAGE_ROTULATION = True
    AMOS = False
    
    idx_hist, idx_co_matrix, idx_lbp, idx_cnn = 0, 1, 2, 3
    
    if MODEL_ACTION == TEST_ALL_CLASSIFIERS:

        idx_xs = [idx_co_matrix]    # features que serão utilizados no treinamento
        xs, y = get_dataset_train()
        x = concatenate_features_by_index(xs, idx_xs)

        x = scale(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2)
        execute_classification(x_train, y_train, x_val, y_val)
        
    elif MODEL_ACTION == TEST_BIG_CLASSIFIER:
        # Monta big classificador conforme trabalho
        
        # CLASSIFICADOR - FEATURES - F1 SCORE INICIAL
        # Neural Net - Co-co - 0.9218462892
        # Random Forest - hist + Co-co - 0.9201626166
        # Neural Net - Resnet50 - 0.9573471487
        # Logistic Regression - Resnet - 0.9524445936
        # Random Forest - LBP + Co-co - 0.9136332857
        # Neural Net - LBP + Co-co + Hist - 0.9228364324
        # Linear SVM - LBP + Co-co + Hist - 0.9015906786
       
        xs, y = get_dataset_train()
        for i in range(len(xs)):
            if len(xs[i]) != 0:
                xs[i] = scale(xs[i])
        
        x_train, x_val, y_train, y_val = train_test_split(list(range(len(y))), y, test_size=.2)
        meta_x_train = []
        meta_x_val = []

        # classificadores do big classificador
        classifiers = [
            MLPClassifier(max_iter=500),
            RandomForestClassifier(n_jobs=-1, n_estimators=100),
            MLPClassifier(max_iter=500),
            LogisticRegression(),
            MLPClassifier(max_iter=500),
            LogisticRegression(),
            RandomForestClassifier(n_jobs=-1, n_estimators=100),
            MLPClassifier(max_iter=500),
            SVC(kernel="linear", C=0.025),
        ]

        # indice dos features por classificador
        idx_xs = [
            [idx_co_matrix],
            [idx_hist, idx_co_matrix],
            [idx_cnn],
            [idx_cnn],
            [idx_cnn],
            [idx_cnn],
            [idx_co_matrix], # , idx_lbp
            [idx_hist, idx_co_matrix], # , idx_lbp
            [idx_hist, idx_co_matrix] # , idx_lbp
        ]

        # Treina, realiza cross-validation, predição, calcula f1, monta o meta x e salva o 
        # classificador treinado 
        treined_clfs = []
        sum_f1 = 0
        for i in range(len(classifiers)):
            x = concatenate_features_by_index(xs, idx_xs[i])
            clf = classifiers[i].fit(split_dataset_by_index(x_train, x), y_train)
            sum_f1 += predict_score(clf, split_dataset_by_index(x_val, x), y_val)
            meta_x_train = mounth_meta_x(clf, meta_x_train, split_dataset_by_index(x_train, x))
            meta_x_val = mounth_meta_x(clf, meta_x_val, split_dataset_by_index(x_val, x))
            treined_clfs.append( clf )
        
        x_deep = load_dataset_tensors()
        deep_model = get_deep_model()
        pred_train = util.np.argmax( deep_model.predict(split_dataset_by_index(x_train, x_deep)), axis=1 )
        pred_val = util.np.argmax( deep_model.predict(split_dataset_by_index(x_val, x_deep)), axis=1 )
        predict_score(None, y=y_val, pred=pred_val)
        meta_x_train = mounth_meta_x(None, meta_x_train, x=None, pred=pred_train)
        meta_x_val = mounth_meta_x(clf, meta_x_val, x=None, pred=pred_val)
            

        # Treina classificadores sobre o meta x
        clf_final = execute_classification(meta_x_train, y_train, meta_x_val, y_val)

        # Realiza a rotução iterativa de novas imagens
        if IMAGE_ROTULATION:
            index_csv = 5
            if AMOS:
                index_csv = 5
            x_unclassifier, _ = get_dataset_train(index_csv=index_csv, load_label=False)
            for i in range(len(x_unclassifier)):
                if len(x_unclassifier[i]) != 0:
                    x_unclassifier[i] = scale(x_unclassifier[i])

            final_pred = []
            for i in range(len(treined_clfs)):
                x = concatenate_features_by_index(x_unclassifier, idx_xs[i])
                pred = treined_clfs[i].predict(x)
                pred = pred.reshape(len(pred), 1)
                final_pred = concatenate_matrix(final_pred, pred)

            x_deep = load_dataset_tensors(True)
            pred = util.np.argmax( deep_model.predict(x_deep), axis=1 )
            final_pred = concatenate_matrix(final_pred, pred)            

            finalmente = clf_final.predict(final_pred)
            y_true = pdn.get_next_labels(pdn.get_all_files_nexet()[0:10000])

            print("acc:", accuracy_score(y_true, finalmente), "f1:", f1_score(y_true, finalmente, average='weighted'))


            '''
            classes = []
            for i in range(len(final_pred)):
                util.printProgressBar(i, len(final_pred))
                counts = util.np.bincount(final_pred[i])
                index = counts.argmax()
                if counts[index] >= len(treined_clfs): # int(sum_f1):
                    classes.append(index)
                else:
                    classes.append(-1)
            
            if AMOS:
                # values = util.get_amos_files()[0:10000]
                values = util.load_csv_file("output/news.csv").values
            else:
                # values = pdt.get_unclassified_imagens()
                values = pdn.get_all_files_nexet()[0:10000]
            values = util.np.append(util.np.array(values).reshape(len(values), 1), util.np.array(classes).reshape(len(classes), 1), axis=1)
            if AMOS:
                pdt.copy_classified_image_AMOS(values)
            else:
                # pdt.copy_classified_image(values)
                pdn.copy_classified_image(values)'''

    elif False:
        # treina o melhor algoritimo, faz a predição de cada imagem e salva na pasta correspondente
        
        x, y = get_dataset()
        x = scale(x)
        # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2)
        
        clf = MLPClassifier()
        clf.fit(x, y)

        x_unclass, names = get_unclassifier_dataset()
        y_unclass = clf.predict(x_unclass)

        values = util.np.append(util.np.array(names).reshape(len(names), 1), 
                            y_unclass.reshape(len(y_unclass),1), axis=1)
        pdt.copy_classified_image(values)
