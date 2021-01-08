#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:57:10 2021

@author: Sope
"""
import numpy as np
from data_gen import data
import uuid
import json

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

train, validation, test = data(uuid.uuid4(), label_form="outcome_pos")
train_data_stats = characterize_data(train)
train_data_stats_test = json.dumps(train_data_stats)