#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:33:04 2021

@author: Sope
"""

from config import config
from db import db, XResult, CalculatedResult
from sqlalchemy.sql import func
import json
import sys
import evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from numpy import array
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

with open("xhyperparameters.json") as f:
    parameters = json.load(f)
    parameters = explode_parameters(parameters)

description_t1 = "hyperparameter_search_t1"
description_t2 = "hyperparameter_search"
split = "84a64c17-fe3e-440c-aaaf-e1bd5b02576f"
t1 = []
t2 = []
for hyperparameter in parameters:
    json_hyperparameter = json.dumps(hyperparameter)
    comparison_t1 = db.session.query(func.avg(CalculatedResult.averaged_test_accuracy)).filter(CalculatedResult.description == description_t1, CalculatedResult.hyperparameters == json_hyperparameter, XResult.input_form == "t1", XResult.split == split).scalar()
    point_t1 = {
        "dropout": hyperparameter["dropout"],\
        "optimizer": hyperparameter["optimizer"],\
        "freeze": hyperparameter["convnet-freeze-percent"],\
        "comparison": comparison_t1,
    }
    t1.append(point_t1)
    comparison_t2 = db.session.query(func.avg(CalculatedResult.averaged_test_accuracy)).filter(CalculatedResult.description == description_t2, CalculatedResult.hyperparameters == json_hyperparameter, XResult.input_form == "t2", XResult.split == split).scalar()
    point_t2 = {
        "dropout": hyperparameter["dropout"],\
        "optimizer": hyperparameter["optimizer"],\
        "freeze": hyperparameter["convnet-freeze-percent"],\
        "comparison": comparison_t2,
    }
    t2.append(point_t2)
    
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
t1_df = pd.DataFrame.from_dict(t1)
t1_df = t1_df.pivot_table( index='optimizer', columns='dropout', values='comparison' )
p1 = sns.heatmap(t1_df, ax=ax1)
ax1.set_title('T1')

t2_df = pd.DataFrame.from_dict(t2)
t2_df = t2_df.pivot_table( index='optimizer', columns='dropout', values='comparison' )
p2 = sns.heatmap(t2_df, ax=ax2)
ax2.set_title('T2')
