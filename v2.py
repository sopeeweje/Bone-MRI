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
from clr_callback import *

from config import config
from data_gen import data, INPUT_FORM_PARAMETERS
from my_optimizers import OPTIMIZERS

MODEL_NAME = "v2"

def apply_layer_freeze(convnet, percent=0.0):
    trainable_layers = [l for l in convnet.layers if len(l.trainable_weights) > 0]
    number_to_freeze = int(percent * len(trainable_layers))
    for i, l in enumerate(trainable_layers):
        if i < number_to_freeze:
            l.trainable = False

def model(input_form="all", aux_size=0, hyperparameters=dict()):
    print("using the following hyperparameters: {}".format(hyperparameters))

    if input_form == "features":
        return features_model(aux_size, hyperparameters)

    parameters = INPUT_FORM_PARAMETERS[input_form]

    inputs = list()
    outputs = list()

    #retreiving the hyperparameters
    DROPOUT = hyperparameters.get("dropout", 0.5) #get "dropout" entry, else return 0.5
    OPTIMIZER = hyperparameters.get("optimizer", "sgd-0001-0.9")
    DEEP_DENSE_TOP = hyperparameters.get("deep-dense-top", True)
    CONVNET_FREEZE_PERCENT = hyperparameters.get("convnet-freeze-percent", 0.0)

    #skip for now
    if parameters["t2"]:
        # init EffNet
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
        # init EffNet
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
        # init EffNet
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
    out = Dropout(rate=DROPOUT)(out)

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
        # define two sets of inputs
        agesex_input = Input(shape=(2,)) #age, sex
        location_input = Input(shape=(14,)) #location
        inputs.append(agesex_input)
        inputs.append(location_input)
        reg=l1_l2(l1=0.00, l2=0.01)
        
        # age and sex network branch
        agesex_1 = Dense(20, kernel_regularizer=reg, activation="relu")(agesex_input)
        agesex_1a = Dropout(rate=0.5)(agesex_1)
        agesex_2 = Dense(20, kernel_regularizer=reg, activation="relu")(agesex_1a)
        agesex_2a = Dropout(rate=0.5)(agesex_2)
        agesex_3 = Dense(20, kernel_regularizer=reg, activation="relu")(agesex_2a)
        predictions_agesex = Dense(1, kernel_regularizer=reg, activation="sigmoid")(agesex_3)
        agesex = Model(inputs=agesex_input, outputs=predictions_agesex)
        
        # location network branch
        location_1 = Dense(20, kernel_regularizer=reg, activation="relu")(location_input)
        location_1a = Dropout(rate=0.5)(location_1)
        location_2 = Dense(20, kernel_regularizer=reg, activation="relu")(location_1a)
        location_2a = Dropout(rate=0.5)(location_2)
        location_3 = Dense(20, kernel_regularizer=reg, activation="relu")(location_2a)
        predictions_location = Dense(1, kernel_regularizer=reg, activation="sigmoid")(location_3)
        location = Model(inputs=location_input, outputs=predictions_location)
        
        combined = concatenate([agesex.output, location.output])
        out = concatenate([out, combined])

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

def features_model(aux_size, hyperparameters):
    OPTIMIZER = hyperparameters.get("optimizer", "sgd-0001-0.9")
    reg=l1_l2(l1=0.00, l2=0.01)
    #aux_input = Input(shape=(aux_size,), name='aux_input')
    # define two sets of inputs
    agesex_input = Input(shape=(2,)) #age, sex
    location_input = Input(shape=(14,)) #location
    
    # age and sex network branch
    agesex_1 = Dense(20, kernel_regularizer=reg, activation="relu")(agesex_input)
    agesex_1a = Dropout(rate=0.5)(agesex_1)
    agesex_2 = Dense(20, kernel_regularizer=reg, activation="relu")(agesex_1a)
    agesex_2a = Dropout(rate=0.5)(agesex_2)
    agesex_3 = Dense(20, kernel_regularizer=reg, activation="relu")(agesex_2a)
    predictions_agesex = Dense(1, kernel_regularizer=reg, activation="sigmoid")(agesex_3)
    agesex = Model(inputs=agesex_input, outputs=predictions_agesex)
    
    # location network branch
    location_1 = Dense(20, kernel_regularizer=reg, activation="relu")(location_input)
    location_1a = Dropout(rate=0.5)(location_1)
    location_2 = Dense(20, kernel_regularizer=reg, activation="relu")(location_1a)
    location_2a = Dropout(rate=0.5)(location_2)
    location_3 = Dense(20, kernel_regularizer=reg, activation="relu")(location_2a)
    predictions_location = Dense(1, kernel_regularizer=reg, activation="sigmoid")(location_3)
    location = Model(inputs=location_input, outputs=predictions_location)
    
    # combine the output of the two branches
    combined = concatenate([agesex.output, location.output])
    
    # apply a FC layer and then a regression prediction on the combined outputs
    z = Dense(2, activation="relu")(combined)
    final_output = Dense(1, kernel_regularizer=reg, activation="sigmoid")(z)
    
    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[agesex.input, location.input], outputs=final_output)
    model.compile(
        loss="binary_crossentropy",
        optimizer=OPTIMIZERS[OPTIMIZER](),
        metrics=["accuracy"])
    return model

def class_weight(training):
    unique, counts = np.unique(training.classes, return_counts=True)
    raw_counts = dict(zip(unique, counts))
    return { l: len(training.classes)/v for l, v in raw_counts.items() }

def train(model, training, validation, run_id, monitor, hyperparameters):
    # callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(
            config.MODEL_DIR,
            "{}-{}.h5".format(
                str(run_id),
                MODEL_NAME,
            ),
        ),
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1,
    )
    early = EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=config.PATIENCE,
        verbose=1,
        mode='auto',
    )
    # Cyclic learning rate
    clr = CyclicLR(base_lr=0.0001, max_lr=0.001, step_size=8*76*2, mode='triangular') #mode='exp_range', gamma=0.99995) # mode='triangular')


    # Train the model - fit_generator from keras
    history = model.fit_generator(
        training,
        steps_per_epoch=training.n / config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=validation,
        validation_steps=math.ceil(validation.n / config.BATCH_SIZE),
        class_weight=class_weight(training),
        callbacks=[checkpoint, early], #[checkpoint, early, sms] 
    )
    return history.history


# first called by the main run function
def run(run_id=None, mode='normal', loaded_data=None, split_id=None, input_form=config.INPUT_FORM,  label_form="outcome", hyperparameters=dict()):
    if run_id is None:
        run_id = int(datetime.utcnow().timestamp())
    if split_id is None:
        split_id = run_id

    if mode == 'normal':
        if loaded_data is None:
            # create the data objects
            training, validation, test = data(split_id, input_form=input_form, label_form=label_form)
        else:
            training, validation, test = loaded_data
        model_instance = model(input_form, aux_size=training.features_size, hyperparameters=hyperparameters)
        # return trained model
        return train(model_instance, training, validation, run_id, 'val_loss', hyperparameters=hyperparameters)
    
    elif mode == 'cross':
        # training, validation, test, holdout_test = loaded_data
        training, validation, test = loaded_data
        model_instance = model(input_form, aux_size=training.features_size, hyperparameters=hyperparameters)
        return train(model_instance, training, validation, run_id, 'val_loss', hyperparameters=hyperparameters)


if __name__ == '__main__':
    run()
