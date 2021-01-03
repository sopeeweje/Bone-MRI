#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:31:06 2020

@author: Sope
"""

import uuid
import traceback
import os
import numpy as np
import pandas
import nrrd
import glob
import argparse
import random
import csv
from shutil import rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
from segmentation import resize 
from config import config
from uuid import uuid4, UUID
import evaluate
from models import models
import json
from keras import backend as K
from PIL import Image
import cv2

from filenames import IMAGE, SEGMENTATION, T1, T2, T1C

axis_sum = {
    0: (1, 1),
    1: (0, 1),
    2: (0, 0),
}

axis_plane = {
    0: lambda image, plane : image[plane, :, :],
    1: lambda image, plane : image[:, plane, :],
    2: lambda image, plane : image[:, :, plane],
}

def select_slice(image, plane, axis=2):
    image = axis_plane[axis](image, plane)
    return image

def load_image(directory_path):
    nrrds = glob.glob("{}/*.nrrd".format(directory_path), recursive=False)
    if len(nrrds) == 3:
        axial, _ = nrrd.read("{}/axial.nrrd".format(directory_path))
        coronal, _ = nrrd.read("{}/coronal.nrrd".format(directory_path))
        sagittal, _ = nrrd.read("{}/sagittal.nrrd".format(directory_path))
        all_images = [axial, coronal, sagittal]
        return [largest_slice(image, 100, 2) for image in all_images]
    else:
        image, _ = nrrd.read("{}/volume.nrrd".format(directory_path)) #import image nrrd
        return [largest_slice(image, 100, a) for a in (0, 1, 2)]

def largest_slice(image, percentile=100, axis=2):
    if image.ndim == 4:
        print("4dim")
        image = image[0]
    i, j = axis_sum[axis]
    sum_on_plane = image.sum(i).sum(j)
    largest_plane = np.argmax(sum_on_plane) #find largest slice
    image = select_slice(image, largest_plane, axis) #select the largest slide
    #image = np.array(Image.fromarray(image, mode="I;16").resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    #image = resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #resize
    return image

def generate_from_features(df, source="data_dir/pretraining/"):
    for index, row in tqdm(df.iterrows(), total=len(df)): #for each patient
        label = row["outcome"]
        name = row["patientID"]
        image = load_image("{}{}".format(source, index))
        yield image, label, name

def relist(l):
    l = list(l)
    if len(l) == 0:
        return l
    return [[k[i] for k in l] for i, _ in enumerate(l[0])]   

class Features(Iterator):
    def __init__(self, features, shuffle, seed):
        super(Features, self).__init__(len(features), config.BATCH_SIZE, shuffle, hash(seed) % 2**32 )
        self.features = np.array(features)
            
    def _get_batches_of_transformed_samples(self, index_array):
        return self.features[index_array] #get features of given batch

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array) 

class Dataset(object):
    def __init__(self, images, labels, names, augment=False, shuffle=False, seed=None):
        self.shuffle = shuffle
        self.augment = augment
        self.names = names
        self.labels = labels
        self.seed = seed

        self.n = len(labels)

        unique, index, inverse, counts = np.unique(self.labels, return_index=True, return_inverse=True, return_counts=True)
        self.y = inverse
        self.classes = inverse
        self.class_indices = { u: i for i, u in enumerate(unique) }

        separate_images = list(zip(*images))
        self.images = np.moveaxis(np.array(separate_images), 0, -1)
        self.datagen = self._get_data_generator()
        self.reset()

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.generator = self.datagen.flow(
                x=self.images,
                y=self.y,
                batch_size=config.BATCH_SIZE,
                shuffle=self.shuffle,
                seed=hash(self.seed) % 2**32,
                )
        self.labels_generator = Features(self.y, self.shuffle, self.seed)

    def _get_data_generator(self):
        if self.augment:
            return ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
            )
        return ImageDataGenerator(
            rescale=1. / 255,
            )

    def next(self):
        labels = self.labels_generator.next()
        inputs = list()
        inputs.append(self.generator.next()[0])
        if len(inputs) == 1:
            inputs = inputs[0]
        return (inputs, labels)
    
    def get_names(self):
        return self.names

def run(model, split_id=None, loaded_data=None, hyperparameters=dict()):
    history = model.run(mode='normal', loaded_data=loaded_data, hyperparameters=hyperparameters)
    model_instance = evaluate.load(os.path.join(config.MODEL_DIR,"pretrained.h5"))
    return history, model_instance

def explode_parameters(parameters):
    all_parameters = []
    for p in parameters.keys():
        if type(parameters[p]) is list:
            for value in parameters[p]:
                new_parameters = dict(parameters)
                new_parameters[p] = value
                all_parameters += explode_parameters(new_parameters)
            break
    if all_parameters:
        return all_parameters
    return [parameters]

# Load training and validation data as dataframes
train = pandas.read_csv("data_dir/pretraining/pretraining_features_train.csv")
validation = pandas.read_csv("data_dir/pretraining/pretraining_features_validation.csv")

# Load images, labels, and names
train_images, train_labels, train_names = relist(generate_from_features(train))
validation_images, validation_labels, validation_names = relist(generate_from_features(validation))

# Load generators
train_generator = Dataset(
            train_images,
            train_labels,
            train_names,
            augment=True,
            shuffle=True
        )
validation_generator = Dataset(
            validation_images,
            validation_labels,
            validation_names,
            augment=False,
            shuffle=False
        )

# Load hyperparameters
with open("hyperparameters.json") as f:
    parameters = json.load(f)
    parameters = explode_parameters(parameters)

# Run model
model = models["pretrain"]
for hyperparameters in parameters:
    history, model_instance = run(model, loaded_data=(train_generator, validation_generator), hyperparameters=hyperparameters)
    K.clear_session()
    
train_generator.reset()
validation_generator.reset()
