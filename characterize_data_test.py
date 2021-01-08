#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:57:10 2021

@author: Sope
"""
import numpy
from data_gen import data
import uuid
import json

def default(o):
    if isinstance(o, numpy.int32): return int(o)
    if isinstance(o, numpy.float64): return float(o)
    if isinstance(o, numpy.float32): return float(o)
    raise TypeError

def characterize_data(data):
    unique, counts = numpy.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

train, validation, test = data(uuid.uuid4(), label_form="outcome_pos")
train_data_stats = characterize_data(train)
for key in train_data_stats:
    print(isinstance(key, numpy.int64))
print(train_data_stats)
train_data_stats_test = json.dumps(train_data_stats, default=default)