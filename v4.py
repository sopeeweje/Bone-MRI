import os
import numpy as np
import math
from tensorflow.keras import applications, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, concatenate
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.regularizers import l1_l2
import efficientnet.tfkeras as efn

from config import config

MODEL_NAME = "v4"

OPTIMIZERS = {
    "sgd-01-0.9": lambda: optimizers.SGD(lr=0.01, momentum=0.9),
    "sgd-001-0.9": lambda: optimizers.SGD(lr=0.001, momentum=0.9),
    "sgd-0001-0.9": lambda: optimizers.SGD(lr=0.0001, momentum=0.9),
    "sgd-01-0.9-nesterov": lambda: optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    "sgd-001-0.9-nesterov": lambda: optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
    "sgd-0001-0.9-nesterov": lambda: optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    "sgd-e6-0.9-nesterov": lambda: optimizers.SGD(lr=1e-6, momentum=0.9, nesterov=True),
    "adam": lambda: "adam",
    "nadam": lambda: "nadam",
}

def apply_layer_freeze(convnet, percent=0.0):
    trainable_layers = [l for l in convnet.layers if len(l.trainable_weights) > 0]
    number_to_freeze = int(percent * len(trainable_layers))
    for i, l in enumerate(trainable_layers):
        if i < number_to_freeze:
            l.trainable = False

def model(aux_size=0, hyperparameters=dict()):
    print("using the following hyperparameters: {}".format(hyperparameters))

    inputs = list()
    outputs = list()

    #retreiving the hyperparameters
    DROPOUT = hyperparameters.get("dropout", 0.5) #get "dropout" entry, else return 0.5
    OPTIMIZER = hyperparameters.get("optimizer", "sgd-0001-0.9")
    DEEP_DENSE_TOP = hyperparameters.get("deep-dense-top", False)
    CONVNET_FREEZE_PERCENT = hyperparameters.get("convnet-freeze-percent", 0.5)

    convnet = efn.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        pooling='avg'
    )
    apply_layer_freeze(convnet, CONVNET_FREEZE_PERCENT)
    out = convnet.output
    inputs.append(convnet.input)
    outputs.append(out)

    if len(outputs) > 1:
        out = concatenate(outputs)
    else:
        out = outputs[0]

    if DEEP_DENSE_TOP:
        out = Dense(256, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = BatchNormalization()(out)
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
        out = Dense(16, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = BatchNormalization()(out)
        out = Dropout(rate=DROPOUT)(out)

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

def class_weight(training):
    unique, counts = np.unique(training.classes, return_counts=True)
    raw_counts = dict(zip(unique, counts))
    return { k: len(training.classes)/v for k, v in raw_counts.items() }

if __name__ == '__main__':
    model = model()
    model.summary()
