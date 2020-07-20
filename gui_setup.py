#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import math

from vis.visualization import visualize_cam, visualize_saliency, overlay
from vis.utils.utils import load_img, normalize, find_layer_idx
from keras.models import load_model, Model
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score
from sklearn import manifold
import pandas
from config import config
from filenames import IMAGE, SEGMENTATION, T1, T2, T1C
from data_gen import load_from_features

sns.set()

def load(filepath):
    return load_model(filepath)

def get_results(model, data):
    results = model.predict_generator(data, steps=math.ceil(len(data)/config.BATCH_SIZE))
    data.reset()
    return results

def transform_binary_probabilities(results):
    probabilities = results.flatten()
    return probabilities

def transform_binary_predictions(results):
    predictions = 1 * (results.flatten() > 0.5)
    return predictions

def get_labels(data):
    return data.classes

t1_image_file = os.path.join(source, "{}-{}-{}".format(index, T1, IMAGE))
t1_seg_file = os.path.join(source, "{}-{}-{}".format(index, T1, SEGMENTATION))
t2_image_file = os.path.join(source, "{}-{}-{}".format(index, T2, IMAGE))
t2_seg_file = os.path.join(source, "{}-{}-{}".format(index, T2, SEGMENTATION))

t1_masked = load_image(t1_image_file, t1_seg_file, verbose=verbose)
t2_masked = load_image(t2_image_file, t2_seg_file, verbose=verbose)

labels, features, name = get_label_features(row, label=label_form)
images, features, labels = input_data_form(t1_masked, t2_masked, t1c_masked, features, labels, input_form=input_form)
yield images, features, labels, name

images, features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))