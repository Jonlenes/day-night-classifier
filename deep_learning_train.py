from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from time import time
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

import util
import keras.applications as app
from vgg16_places_365 import VGG16_Places365, preprocess_input as pi
import numpy as np


def load_dataset_tensors():
    """
    Carrega as imagens e converte para tenors para treinar a deep
    """
    files1 = util.get_all_files_names("output/Dia/")
    files2 = util.get_all_files_names("output/Noite/")
    files3 = util.get_all_files_names("output/Trasnsicao/")

    files_names = files1 + files2 + files3
    labels = [0] * len(files1) + [1] * len(files2) + [2] * len(files3)

    files_train, files_val, targets_train, targets_valid = train_test_split(files_names, labels, test_size=.2)

    train_tensors = util.paths_to_tensor(files_train)
    train_tensors = preprocess_input(train_tensors)

    valid_tensors = util.paths_to_tensor(files_val)
    valid_tensors = preprocess_input(valid_tensors)
    
    targets_train = np_utils.to_categorical(util.np.array(targets_train), 3)
    targets_valid = np_utils.to_categorical(util.np.array(targets_valid), 3)

    return train_tensors, targets_train, valid_tensors, targets_valid
 

def pre_trained_model(index):
    """
    Modelos que foi realizado o fine tunning
    """
    if index == 1:                   
        return ["VGG16_Places365", VGG16_Places365(include_top=False, weights='places', input_shape=(224, 224, 3)), pi]    
    elif index == 0:
        return ["ResNet50", ResNet50(weights='imagenet', include_top=False), app.resnet50.preprocess_input]
    

def finetuned_model2(x_train, y_train, valid_tensors, targets_valid, base_model, epochs, batch_size, index_model):
    """
    Realiza o fine tunning das CNNs
    """
    # get layers and add average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if index_model == 0: #ResNet50
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)

        # x = Dropout(0.25)(x)
        # x = Dense(1024, activation='relu', name='fc-2')(x)
        # x = Dropout(0.5)(x)
        
        # a softmax layer for 4 classes
        # x = Dense(128, activation='softmax', name='output_layer')(x)

        '''
        # add fully-connected layer
        #x = Dense(512, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # add fully-connected & dropout layers
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        # x = Dropout(0.25)(x)
        # x = Dropout(0.5)(x)
        '''
    else: # VGG16
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
    # add output layer
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in base_model.layers:
        layer.trainable = False

    # update the weight that are added
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train)

    # choose the layers which are updated by training
    layer_num = len(model.layers)
    perct_freezed = 0.9

    for layer in model.layers[:int(layer_num * perct_freezed)]:
        layer.trainable = False

    for layer in model.layers[int(layer_num * perct_freezed):]:
        layer.trainable = True

    # update the weights
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="saved_models/weights.best." + str(index_model) + ".hdf5", verbose=1, save_best_only=True)
    model.fit(x_train, y_train, validation_data=(valid_tensors, targets_valid), epochs=epochs, callbacks=[checkpointer], batch_size=batch_size)

    return model

    
def finetuned_model(x_train, y_train, valid_tensors, targets_valid, base_model, epochs, batch_size, index_model):
 
    last_layer = base_model.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(3, activation='softmax')(x)
    custom_resnet_model2 = Model(inputs=base_model.input, outputs=out)

    custom_resnet_model2.summary()
 
    for layer in custom_resnet_model2.layers[:-6]:
        layer.trainable = False
 
    custom_resnet_model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    t = time()
    checkpointer = ModelCheckpoint(filepath="saved_models/weights.best." + str(index_model) + ".hdf5", verbose=1, save_best_only=True)
    custom_resnet_model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                            verbose=1, validation_data=(valid_tensors, targets_valid), 
                            callbacks=[checkpointer])
    print('Training time: %s' % (time() - t))
    return custom_resnet_model2


if __name__ == "__main__":

    # train_tensors, targets_train, valid_tensors, targets_valid = load_dataset_tensors()

    basic_model = False
    data_augmentation = False
    epochs = 5
    batch_size = 20   #Não está sendo usado no momento
    # n_examples = train_tensors.shape[0]
    
    print(epochs, batch_size)
    
    for i in [0]:
        name, model, preprocess = pre_trained_model(i)
        print(name)

        train_tensors, targets_train, valid_tensors, targets_valid = load_dataset_tensors()
        train_tensors = preprocess(train_tensors)
        valid_tensors = preprocess(valid_tensors)

        tried_model = finetuned_model(train_tensors, targets_train, valid_tensors, targets_valid, model, epochs, batch_size, i)
        
        # get index of predicted dog breed for each image in test set
        predictions = tried_model.predict(valid_tensors)
        labels = np.argmax(predictions, axis=1)
        print(predictions.shape, labels.shape)

        # report test accuracy
        test_accuracy = 100 * np.sum(np.array(labels)==np.argmax(targets_valid, axis=1))/len(labels)
        print('Test accuracy: %.4f%%' % test_accuracy)

        del name, model, preprocess, tried_model, predictions, labels, test_accuracy, train_tensors, targets_train, valid_tensors, targets_valid
