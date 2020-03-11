import os
import numpy as np
import math
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, concatenate
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
import efficientnet.keras as efn
import datetime as datetime


from config import config
from data_gen import data, INPUT_FORM_PARAMETERS

batch_size = [1, 5, 16, 32]
epochs = [100, 300, 500]
dropout_rate = [0, 0.1, 0.5]
lr = [0.01, 0.001, 0.0001, 0.00001]
momentum = [0.8, 0.85, 0.9]
optimizer = ["SGD","adam"]

def apply_layer_freeze(convnet, percent=0.0):
    trainable_layers = [l for l in convnet.layers if len(l.trainable_weights) > 0]
    number_to_freeze = int(percent * len(trainable_layers))
    for i, l in enumerate(trainable_layers):
        if i < number_to_freeze:
            l.trainable = False

def model(input_form="all", aux_size=0, hyperparameters=dict()):
    print("using the following hyperparameters: {}".format(hyperparameters))
    parameters = INPUT_FORM_PARAMETERS[input_form]

    inputs = list()
    outputs = list()

    #retreiving the hyperparameters
    DROPOUT = hyperparameters.get("dropout", 0.5) #get "dropout" entry, else return 0.5
    OPTIMIZER = hyperparameters.get("optimizer", "sgd-0001-0.9")
    DEEP_DENSE_TOP = hyperparameters.get("deep-dense-top", True)
    CONVNET_FREEZE_PERCENT = hyperparameters.get("convnet-freeze-percent", 0.0)

    if parameters["t2"]:
        convnet = efn.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        for layer in convnet.layers:
            layer.name = "{}_t2".format(layer.name)
        apply_layer_freeze(convnet, CONVNET_FREEZE_PERCENT)
        out = convnet.output
        out = Flatten()(out)
        inputs.append(convnet.input)
        outputs.append(out)

    if parameters["t1"]:
        convnet = efn.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        for layer in convnet.layers:
            layer.name = "{}_t1".format(layer.name)
        apply_layer_freeze(convnet, CONVNET_FREEZE_PERCENT)
        out = convnet.output
        out = Flatten()(out)
        inputs.append(convnet.input)
        outputs.append(out)
        
    if parameters["t1c"]:
        convnet = efn.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        for layer in convnet.layers:
            layer.name = "{}_t1c".format(layer.name)
        apply_layer_freeze(convnet, CONVNET_FREEZE_PERCENT)
        out = convnet.output
        out = Flatten()(out)
        inputs.append(convnet.input)
        outputs.append(out)

    if len(outputs) > 1:
        out = concatenate(outputs)
    else:
        out = outputs[0]

    out = Dense(256, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
    out = BatchNormalization()(out)

    if DEEP_DENSE_TOP:
        out = Dropout(rate=DROPOUT)(out)
        out = Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = BatchNormalization()(out)
        out = Dropout(rate=DROPOUT)(out)
        out = Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = BatchNormalization()(out)
        out = Dropout(rate=DROPOUT)(out)
        out = Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = BatchNormalization()(out)
        out = Dropout(rate=DROPOUT)(out)

    if parameters["features"]:
        aux_input = Input(shape=(aux_size,), name='aux_input')
        inputs.append(aux_input)
        out = concatenate([out, aux_input])

    out = Dense(16, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
    out = BatchNormalization()(out)
    predictions = Dense(1, activation="sigmoid", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)

    # creating the final model
    if len(inputs) > 1:
        model = Model(inputs=inputs, outputs=predictions)
    else:
        model = Model(inputs=inputs[0], outputs=predictions)

    # compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=OPTIMIZERS[OPTIMIZER](),
        metrics=["accuracy"])

    return model