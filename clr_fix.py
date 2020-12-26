import argparse
import os
import math
import json

from datetime import datetime
from models import models
from db import db, Result
from uuid import uuid4, UUID

from keras import backend as K

import numpy as np
import evaluate
from data_gen import data

from config import config

def test_model(model, train, validation, test, label_form):

    loss, accuracy = model.evaluate_generator(validation, steps=math.ceil(len(validation)/config.BATCH_SIZE))
    train_loss, train_accuracy = model.evaluate_generator(train, steps=math.ceil(len(train)/config.BATCH_SIZE))
    test_loss, test_accuracy = model.evaluate_generator(test, steps=math.ceil(len(test)/config.BATCH_SIZE))

    train.reset()
    validation.reset()
    test.reset()

    results = evaluate.get_results(model, validation)
    labels = list(evaluate.get_labels(validation))

    test_results = evaluate.get_results(model, test)
    test_labels = list(evaluate.get_labels(test))
    
    if label_form == "outcome_3":
        probabilities = list(results)
        test_probabilities = list(test_results)
    else:
        probabilities = list(evaluate.transform_binary_probabilities(results))
        test_probabilities = list(evaluate.transform_binary_probabilities(test_results))

    train.reset()
    validation.reset()
    test.reset()

    return {
        "train_accuracy": float(train_accuracy),
        "train_loss": float(train_loss),
        "accuracy": float(accuracy),
        "loss": float(loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "probabilities": probabilities,
        "labels": labels,
        "test_probabilities": test_probabilities,
        "test_labels":test_labels,
    }

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

def run(model, description, input_form, label_form="outcome", split_id=None, loaded_data=None, hyperparameters=dict()):

    model_instance = evaluate.load(model)

    if loaded_data is None:
        train, validation, test = data(split_id, input_form=input_form, label_form=label_form)
    else:
        train, validation, test = loaded_data
        train.reset()
        validation.reset()
        test.reset()

    train_data_stats = characterize_data(train)
    validation_data_stats = characterize_data(validation)
    test_data_stats = characterize_data(test)
    results = test_model(model_instance, train, validation, test, label_form)

    result = Result(
        "v2",
        str(uuid4())
,
        str(split_id),
        train_data_stats,
        validation_data_stats,
        test_data_stats,
        description,
        input_form,
        label=label_form,
        hyperparameters=hyperparameters,
        history="-",
        **results
        )
    db.session.add(result)
    db.session.commit()

run("output/models/fc2dc27b-6954-4f82-80ca-0973d9437eb3-v2.h5", 
    description = "B0_clr", 
    input_form = "t2",
    label_form = "outcome_pos",
    split_id = UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"))
