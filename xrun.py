import argparse
import os
import math
import json
import numpy

from datetime import datetime
from models import models
from db import db, XResult
from uuid import uuid4, UUID

from keras import backend as K

import numpy as np
import evaluate
from data import xdata

from config import config

import pandas
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import xanalyze


def test_model(model, train, validation, test):  # , holdout_test):

    train_loss, train_accuracy = model.evaluate_generator(train, steps=math.ceil(len(train) / config.BATCH_SIZE))
    loss, accuracy = model.evaluate_generator(validation, steps=math.ceil(len(validation)/config.BATCH_SIZE))
    test_loss, test_accuracy = model.evaluate_generator(test, steps=math.ceil(len(test)/config.BATCH_SIZE))
    # holdout_test_loss, holdout_test_accuracy = model.evaluate_generator(holdout_test, steps=math.ceil(len(holdout_test)/config.BATCH_SIZE))

    train.reset()
    validation.reset()
    test.reset()
    # holdout_test.reset()

    # labels - ground truths
    # results - predicted results from model
    results = evaluate.get_results(model, validation)
    probabilities = list(evaluate.transform_binary_probabilities(results))
    labels = list(evaluate.get_labels(validation))

    test_results = evaluate.get_results(model, test)
    test_probabilities = list(evaluate.transform_binary_probabilities(test_results))
    test_labels = list(evaluate.get_labels(test))

    # holdout_test_results = evaluate.get_results(model, holdout_test)
    # holdout_test_probabilities = list(evaluate.transform_binary_probabilities(holdout_test_results))
    # holdout_test_labels = list(evaluate.get_labels(holdout_test))

    train.reset()
    validation.reset()
    test.reset()
    # holdout_test.reset()

    # get binary predictions
    # holdout_binary_predictions = list(evaluate.transform_binary_predictions(holdout_test_results))
    test_binary_predictions = list(evaluate.transform_binary_predictions(test_results))
    # get f1 score
    test_f1_result = f1_score(test_labels, test_binary_predictions)
    # holdout_f1_result = f1_score(holdout_test_labels, holdout_binary_predictions)

    return {
        "train_accuracy": float(train_accuracy),
        "train_loss": float(train_loss),
        "accuracy": float(accuracy),
        "loss": float(loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        # "holdout_test_accuracy": float(holdout_test_accuracy),
        # "holdout_test_loss": float(holdout_test_loss),
        "holdout_test_accuracy": float(0),
        "holdout_test_loss": float(0),
        "probabilities": probabilities,
        "labels": labels,
        "test_probabilities": test_probabilities,
        "test_labels": test_labels,
        # "holdout_test_probabilities": holdout_test_probabilities,
        # "holdout_test_labels": holdout_test_labels,
        "holdout_test_probabilities": 'na',
        "holdout_test_labels": 'na',
        "test_f1_result": test_f1_result,
        # "holdout_f1_result": holdout_f1_result,
        "holdout_f1_result": float(0),
    }

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

def xrun(fold, loaded_data, model, description, input_form, label_form="outcome", split_id=None, hyperparameters=dict()):

    # gen random id for this specific parameter run
    run_id = uuid4()
    if split_id is None:
        split_id = run_id

    # calling run function on the cross validation v1 model
    history = model.run(run_id,  mode='cross', loaded_data=loaded_data, input_form=input_form, label_form=label_form, hyperparameters=hyperparameters)
    # clearing the training session because we will be running multiple models
    K.clear_session()

    # load the model instance - this takes awhile
    model_instance = evaluate.load(os.path.join(
        config.MODEL_DIR,
        "{}-{}.h5".format(str(run_id), model.MODEL_NAME),
        ))

    # fold_train, fold_validation, fold_test, fold_holdout_test = loaded_data
    fold_train, fold_validation, fold_test = loaded_data

    fold_train.reset()
    fold_validation.reset()
    fold_test.reset()
    # fold_holdout_test.reset()

    train_data_stats = characterize_data(fold_train)
    validation_data_stats = characterize_data(fold_validation)
    test_data_stats = characterize_data(fold_test)
    # holdout_test_data_stats = characterize_data(fold_holdout_test)

    # testing of the data
    # results = test_model(model_instance, fold_train, fold_validation, fold_test, fold_holdout_test)
    results = test_model(model_instance, fold_train, fold_validation, fold_test)
    fold_train.reset()
    fold_validation.reset()
    fold_test.reset()
    # fold_holdout_test.reset()

    holdout_test_data_stats = 'na'

    result = XResult(
        fold,
        config.NUMBER_OF_FOLDS,
        model.MODEL_NAME,
        str(run_id),
        str(split_id),
        train_data_stats,
        validation_data_stats,
        test_data_stats,
        holdout_test_data_stats,
        description,
        input_form,
        label_form,
        hyperparameters=hyperparameters,
        history=history,
        **results
        )
    db.session.add(result)
    db.session.commit()


# helper for setting up parameters
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='which model to run (see models.py)')
    parser.add_argument(
        '--description',
        type=str,
        help='brief description of the run and its differences')
    parser.add_argument(
        '--form',
        type=str,
        help='input form (see data.py for more information)',
        default=config.INPUT_FORM,
        )
    parser.add_argument(
        '--label',
        type=str,
        help='label form (see data.py for more information)',
        default="outcome",
        )
    parser.add_argument(
        '--split',
        type=str,
        help='seed for split',
        default=None,
        )
    parser.add_argument(
        '--hyperparameters',
        type=str,
        help='hyperparameters file',
        required=True,
        )
    parser.add_argument(
        '--trials',
        type=int,
        default=config.TRIALS,
        help='how many times to run')
    FLAGS, unparsed = parser.parse_known_args()
    with open(FLAGS.hyperparameters) as f:
        parameters = json.load(f)
        parameters = explode_parameters(parameters)
    model = models[FLAGS.model]
    split = FLAGS.split
    if split is None:
        split = uuid4()
    else:
        split = UUID(split)
    # check to make sure we didn't use the description on another run AND split
    # xanalyze.check_run_and_split(FLAGS.description, str(split), FLAGS.form)

    # splitting the initial training and holdout test sets
    f = pandas.read_pickle(config.FEATURES)

    put_In_Training = pandas.read_csv(config.MULTIPLE_LESIONS)
    df_List = list(put_In_Training['ID'])
    multiple = f[f['patient'].isin(df_List)]
    multiple_y = multiple[FLAGS.label].values

    new_df = f[~f.patient.isin(df_List)]
    y = new_df[FLAGS.label].values

    # set up the k-fold process
    skf = StratifiedKFold(n_splits=config.NUMBER_OF_FOLDS, random_state=int(split) % 2 ** 32)

    # get the folds and loop over each fold
    fold_number = 0
    for train_index, test_index in skf.split(new_df, y):
        fold_number += 1
        # get the training and testing set for the fold
        X_train, testing = new_df.iloc[train_index], new_df.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #append multiple lesions into training/validation
        X_train = X_train.append(multiple, ignore_index=False)
        y_train = numpy.concatenate((y_train, multiple_y))

        # split the training into training and validation
        training, validation, result_train, result_test = train_test_split(X_train, y_train, test_size=config.SPLIT_TRAINING_INTO_VALIDATION, stratify=y_train, random_state=int(split) % 2 ** 32)

        # get the data
        # training_data, validation_data, testing_data, holdout_test_data = xdata(fold_number, training, validation, testing, holdout_test, split, input_form=FLAGS.form, label_form=FLAGS.label)
        training_data, validation_data, testing_data = xdata(fold_number, training, validation, testing, split, input_form=FLAGS.form, label_form=FLAGS.label)
        # run the training, each trial
        for _ in range(FLAGS.trials):
            # in each trial, run for each hyperparameter combination
            for hyperparameters in parameters:
                # xrun(fold_number, (training_data, validation_data, testing_data, holdout_test_data), model, FLAGS.description, FLAGS.form, FLAGS.label, split, hyperparameters=hyperparameters)
                xrun(fold_number, (training_data, validation_data, testing_data), model,
                     FLAGS.description, FLAGS.form, FLAGS.label, split, hyperparameters=hyperparameters)
                K.clear_session()

    # independent testing across trials * folds
    print('The split id for this run ' + FLAGS.description + ' is ' + str(split))
    xanalyze.analyze_averages(parameters, FLAGS.description, FLAGS.model, FLAGS.form, str(split))


