"""
Adapted from keras example cifar10_cnn.py
Train NASNet-CIFAR on the CIFAR10 small images dataset.
"""
from __future__ import print_function
import os

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import numpy as np

from clr import OneCycleLR
from models.small.model import MiniVGG

if not os.path.exists('weights/'):
    os.makedirs('weights/')

weights_file = 'weights/mini_vgg.h5'
model_checkpoint = ModelCheckpoint(
    weights_file,
    monitor='val_acc',
    save_best_only=True,
    save_weights_only=True,
    mode='max')
batch_size = 128
nb_classes = 10
nb_epoch = 50  # Only finding lr
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# preprocess input
mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True).astype('float32')
std = np.mean(X_train, axis=(0, 1, 2), keepdims=True).astype('float32')

print("Channel Mean : ", mean)
print("Channel Std : ", std)

X_train = (X_train - mean) / (std)
X_test = (X_test - mean) / (std)

# Learning rate finder callback setup
num_samples = X_train.shape[0]

# When using the validation set for LRFinder, try out values starting from 2x
# the lr found there and move lower until its good for the first few epochs
lr_manager = OneCycleLR(
    max_lr=0.025,
    end_percentage=0.2,
    scale_percentage=0.1,
    maximum_momentum=0.95,
    verbose=True)

# For training, the auxilary branch must be used to correctly train NASNet
model = MiniVGG((img_rows, img_cols, img_channels),
                weight_decay=1e-5,
                weights=None,
                classes=nb_classes)
model.summary()

# These values will be overridden by the above callback
optimizer = SGD(lr=0.0025, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.load_weights(weights_file)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        shuffle=True,
        verbose=1,
        callbacks=[lr_manager, model_checkpoint])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
    #                     steps_per_epoch=X_train.shape[0] // batch_size,
    #                     validation_data=(X_test, Y_test),
    #                     epochs=nb_epoch, verbose=1,
    #                     callbacks=[lr_manager, model_checkpoint])

scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
for score, metric_name in zip(scores, model.metrics_names):
    print("%s : %0.4f" % (metric_name, score))
