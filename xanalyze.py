from config import config
from db import db, XResult, CalculatedResult
from sqlalchemy.sql import func
import json
import sys
import evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from numpy import array
import xrun


# averaging calculations
def analyze_averages(parameters, description, model, input_form, split):

    for hyperparameter in parameters:
        json_hyperparameter = json.dumps(hyperparameter)

        # AVERAGE TESTING and HOLDOUT results across ALL trials
        averaged_test_f1 = db.session.query(func.avg(XResult.holdout_f1_result)).filter(XResult.description == description, XResult.hyperparameters == json_hyperparameter, XResult.input_form == input_form, XResult.split == split).scalar()
        averaged_test_accuracy = db.session.query(func.avg(XResult.test_accuracy)).filter(XResult.description == description, XResult.hyperparameters == json_hyperparameter, XResult.input_form == input_form, XResult.split == split).scalar()
        averaged_test_loss = db.session.query(func.avg(XResult.test_loss)).filter(XResult.description == description, XResult.hyperparameters == json_hyperparameter, XResult.input_form == input_form, XResult.split == split).scalar()
        # averaged_holdout_f1 = db.session.query(func.avg(XResult.holdout_f1_result)).filter(XResult.description == description, XResult.hyperparameters == json_hyperparameter, XResult.input_form == input_form, XResult.split == split).scalar()
        # averaged_holdout_accuracy = db.session.query(func.avg(XResult.holdout_test_accuracy)).filter(XResult.description == description, XResult.hyperparameters == json_hyperparameter, XResult.input_form == input_form, XResult.split == split).scalar()
        # averaged_holdout_loss = db.session.query(func.avg(XResult.holdout_test_loss)).filter(XResult.description == description, XResult.hyperparameters == json_hyperparameter, XResult.input_form == input_form, XResult.split == split).scalar()

        # Ensembling the k-fold models by minimum loss
        # min_loss_f1, min_loss_accuracy, min_loss_test_acc, min_loss_test_loss = ensemble_folds(json_hyperparameter, description, "min", input_form)
        # Ensembling the k-fold models by maximum accuracy
        # max_acc_f1, max_acc_accuracy, max_acc_test_acc, max_acc_test_loss = ensemble_folds(json_hyperparameter, description, "max", input_form)

        # save it all
        result = CalculatedResult(
            str(split),
            model,
            # min_loss_f1,
            float(0),
            # min_loss_accuracy,
            float(0),
            # min_loss_test_acc,
            float(0),
            # min_loss_test_loss,
            float(0),
            # max_acc_f1,
            float(0),
            # max_acc_accuracy,
            float(0),
            # max_acc_test_acc,
            float(0),
            # max_acc_test_loss,
            float(0),
            averaged_test_f1,
            averaged_test_accuracy,
            averaged_test_loss,
            # averaged_holdout_f1,
            float(0),
            # averaged_holdout_accuracy,
            float(0),
            # averaged_holdout_loss,
            float(0),
            hyperparameter,
            description,
            input_form
        )
        db.session.add(result)
        db.session.commit()
    return


# for each n-fold go through each trial to pick model with LEAST TESTING LOSS for a total of n-final models per hyperparameter
# mode is max for now
# NEED HOLDOUT SET TO RUN THIS
def ensemble_folds(json_hyperparameter, description, mode, input_form):
    least_loss_list = list()
    # get the trials with the least loss for each fold

    if mode == "min":
        for x in range(config.NUMBER_OF_FOLDS):
            subquery = db.session.query(func.min(XResult.test_loss)).filter(XResult.description == description,
                                                                            XResult.hyperparameters == json_hyperparameter,
                                                                            XResult.input_form == input_form,
                                                                            XResult.fold == x + 1)
            model_with_least_loss = db.session.query(XResult).filter(XResult.description == description,
                                                                     XResult.hyperparameters == json_hyperparameter,
                                                                     XResult.input_form == input_form,
                                                                     XResult.fold == x + 1,
                                                                     XResult.test_loss == subquery).first()
            least_loss_list.append(model_with_least_loss)

    if mode == "max":
        for x in range(config.NUMBER_OF_FOLDS):
            subquery = db.session.query(func.max(XResult.test_accuracy)).filter(XResult.description == description,
                                                                                XResult.hyperparameters == json_hyperparameter,
                                                                                XResult.input_form == input_form,
                                                                                XResult.fold == x + 1)
            model_with_most_accuracy = db.session.query(XResult).filter(XResult.description == description,
                                                                        XResult.hyperparameters == json_hyperparameter,
                                                                        XResult.input_form == input_form,
                                                                        XResult.fold == x + 1,
                                                                        XResult.test_accuracy == subquery).first()
            least_loss_list.append(model_with_most_accuracy)

    # average the probabilities of the n models for the n folds selected
    flag = 0
    holdout_labels = list()
    holdout_probs = list()
    test_loss = 0
    test_acc = 0

    for y in least_loss_list:
        fold_holdout_probs = y.get_holdout_probabilities()
        check_holdout_labels = y.get_holdout_labels()
        test_loss = test_loss + y.test_loss
        test_acc = test_acc + y.test_accuracy

        # test to make sure the holdout label list is the same across all trials
        if flag == 0:
            holdout_labels = check_holdout_labels
            holdout_probs = fold_holdout_probs
            flag = 1
        elif holdout_labels != check_holdout_labels:
            sys.stderr.write("Holdout labels for this run: " + description + " did not match!")
            return
        else:
            #add the new probabilities on to average later
            holdout_probs = [x + y for x, y in zip(holdout_probs, fold_holdout_probs)]

    avg_holdout_probs = [i / len(least_loss_list) for i in holdout_probs]

    # convert averaged probabilities to binary
    array_avg_holdout_probs = array(avg_holdout_probs)
    holdout_binary_predictions = list(evaluate.transform_binary_predictions(array_avg_holdout_probs))

    # using the probabilities, test it against the holdout set
    holdout_f1_result = f1_score(holdout_labels, holdout_binary_predictions)
    holdout_accuracy_result = accuracy_score(holdout_labels, holdout_binary_predictions)

    # also return the averaged test accuracy and loss of the selected models
    test_loss = test_loss / len(least_loss_list)
    test_acc = test_acc / len(least_loss_list)

    return holdout_f1_result, holdout_accuracy_result, test_acc, test_loss

# given the run name, split seed, and input_form
def check_run_and_split(description, split, input_form):
    subquery = db.session.query(XResult).filter(XResult.description == description, XResult.split == split, XResult.input_form == input_form)
    check = db.session.query(subquery.exists()).scalar()
    if check:
        sys.stderr.write("The run " + description + " and split " + split + " already exists in the database")
        sys.exit()
    else:
        return


def get_models(split, description, fold):

    model_list = list()

    input_form = ['t1', 't2', 'features']

    for x in input_form:
        subquery = db.session.query(func.max(XResult.accuracy)).filter(XResult.description == description,
                                                                            XResult.split == split,
                                                                            XResult.fold == fold,
                                                                            XResult.input_form == x)

        model_with_most_accuracy = db.session.query(XResult).filter(XResult.description == description,
                                                                    XResult.split == split,
                                                                    XResult.fold == fold,
                                                                    XResult.input_form == x,
                                                                    XResult.accuracy == subquery).first()

        model_list.append(model_with_most_accuracy.run_id)

    return model_list


def get_models(split, description, fold, hyperparameters):

    input_form = ['t1', 't2', 'features']

    with open(hyperparameters) as f:
        parameters = json.load(f)
        parameters = xrun.explode_parameters(parameters)

    for hyperparameter in parameters:
        json_hyperparameter = json.dumps(hyperparameter)
        print(json_hyperparameter)
        for x in input_form:

            subquery = db.session.query(func.max(XResult.accuracy)).filter(XResult.description == description,
                                                                           XResult.split == split,
                                                                           XResult.fold == fold,
                                                                           XResult.input_form == x,
                                                                           XResult.hyperparameters == json_hyperparameter)

            models = db.session.query(XResult).filter(XResult.description == description,
                                                                        XResult.split == split,
                                                                        XResult.fold == fold,
                                                                        XResult.input_form == x,
                                                                        XResult.hyperparameters == json_hyperparameter,
                                                                        XResult.accuracy == subquery).all()


            best = 0
            best_uuid = 'na'
            for s in models:
                sum = round(s.accuracy, 3) + round(s.test_accuracy, 3)
                if sum > best:
                    best = sum
                    best_uuid = s.run_id


            print('"' + best_uuid + '", # ' + x)

    return
