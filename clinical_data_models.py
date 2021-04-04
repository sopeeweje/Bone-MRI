#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:22:00 2020

@author: Sope
"""
from sklearn import tree, preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier, ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
from db import db, Result
from uuid import uuid4, UUID
import pickle
from config import config
from data_gen import data
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

input_data = "features/training_features.csv"
all_data = []

best_acc = 0
best_model = None
best_models = []

def features_data(training_data, validation_data, testing_data, external_data):
    #get data from keras dataset generators
    training_features = training_data.features
    training_labels = training_data.labels
    validation_features = validation_data.features
    validation_labels = validation_data.labels
    testing_features = testing_data.features
    testing_labels = testing_data.labels
    external_features = external_data.features
    external_labels = external_data.labels
    
    #lists for data to be given to logistic regression
    train_set = []
    train_labels = []
    val_set = []
    val_labels = []
    test_set = []
    test_labels = []
    ext_set = []
    ext_labels = []


    #create training data
    for i in range(len(training_features)):
        age = training_features[i][0]
        sex = training_features[i][1]
        location = training_features[i][2] #.strip('][').split(', ') 
        addition = [age, sex]
        for e in location:
            addition.append(e)
        train_set.append(addition)
        train_labels.append(training_labels[i])
    
    #create validation data
    for i in range(len(validation_features)):
        age = validation_features[i][0]
        sex = validation_features[i][1]
        location = validation_features[i][2] #.strip('][').split(', ') 
        addition = [age, sex]
        for e in location:
            addition.append(e)
        val_set.append(addition)
        val_labels.append(validation_labels[i])
        
    #create testing data
    for i in range(len(testing_features)):
        age = testing_features[i][0]
        sex = testing_features[i][1]
        location = testing_features[i][2] #.strip('][').split(', ') 
        addition = [age, sex]
        for e in location:
            addition.append(e)
        test_set.append(addition)
        test_labels.append(testing_labels[i])
        
    #create external data
    for i in range(len(external_features)):
        age = external_features[i][0]
        sex = external_features[i][1]
        location = external_features[i][2] #.strip('][').split(', ') 
        addition = [age, sex]
        for e in location:
            addition.append(e)
        ext_set.append(addition)
        ext_labels.append(external_labels[i])

    train_set = preprocessing.scale(train_set)
    val_set = preprocessing.scale(val_set)
    test_set = preprocessing.scale(test_set)
    ext_set = preprocessing.scale(ext_set)
    
    return train_set, train_labels, val_set, val_labels, test_set, test_labels, ext_set, ext_labels

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { int(c): index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

def features_run(label_form, classifier, split_id=None, model="n/a"):
    #create split id and run id
    run_id = uuid4()
    if split_id is None:
        split_id = run_id 
    
    #create initial data
    training_data, validation_data, testing_data, external_data = data(seed=split_id,
        input_form="features",
        label_form=label_form,
        train_shuffle=False,
        validation_shuffle=False,
        test_shuffle=False,
        train_augment=False,
        validation_augment=False,
        test_augment=False,
        validation_split=config.VALIDATION_SPLIT,
        test_split=config.TEST_SPLIT,
        verbose=True,
        )

    train_set, train_labels, val_set, val_labels, test_set, test_labels, external_set, external_labels = features_data(training_data, validation_data, testing_data, external_data)
    
    for j in range(1,2):
        run_id = uuid4()
        history = []
        best_acc = -100
        best_model = None
        c = classifier
        #model_best = 0
        model_acc = 0
        for i in tqdm(range(1)): 
            #clf = c(random_state=i, **(PARAMETERS[j]))
            clf = c(random_state=i)#, max_depth=depth) #max_iter=1000)#, kernel="linear", probability=True)
            clf.fit(train_set, train_labels)
            clf = RFECV(clf, step=1) #n_features_to_select=j, 
            clf.fit(train_set, train_labels)
            score = clf.score(val_set, val_labels)
            if score > best_acc:
                best_acc = score
                best_model = clf
            if score > model_acc:
                model_acc = score
                #model_best = clf
            history.append(score)
        #print(best_model.score(test_set, test_labels))
        #print(best_model.coef_)
        #print(best_model.score(test_set, test_labels))
        #tree_plot = plt.figure(2)
        #tree.plot_tree(best_model)
        #tree_plot.savefig("featuremodel.png")
        
        probabilities=best_model.predict_proba(val_set).tolist()
        probabilities = [i[1] for i in probabilities]
        test_probabilities=best_model.predict_proba(test_set).tolist()
        test_probabilities = [i[1] for i in test_probabilities]
        
        result = Result(
                model=model,
                uuid=str(run_id),
                split_uuid=str(split_id),
                train_data_stats=characterize_data(training_data),
                validation_data_stats=characterize_data(validation_data),
                test_data_stats=characterize_data(testing_data),
                description=str(best_model),
                input_form="features",
                label=label_form,
                hyperparameters="",
                history="",
                train_accuracy=best_model.score(train_set, train_labels),
                train_loss=0,
                accuracy=best_model.score(val_set, val_labels),
                loss=j,
                test_accuracy=best_model.score(test_set, test_labels),
                test_loss=0,
                probabilities=probabilities,
                labels=best_model.predict(val_set).tolist(),
                test_probabilities=test_probabilities,
                test_labels=best_model.predict(test_set).tolist(),
                )
    
        history = db.Column(db.String)
        db.session.add(result)
        db.session.commit()
        
        filename = '{}/models/{}_features.sav'.format(config.OUTPUT, str(run_id))
        pickle.dump(best_model, open(filename, 'wb'))
        
        print("Number of Features: {}, Validation AUC: {}".format(str(j), str(roc_auc_score(val_labels, probabilities))))
        print("Rankings: {}".format(best_model.ranking_))
        #print("Coefficients: {}".format(best_model.estimator.feature_importances_))
        print("Coefficients: {}".format(best_model.importance_getter))
        print("")
if __name__ == '__main__':
    #features_run("outcome_pos", LogisticRegression, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "logistic regression")
    features_run("outcome_pos", GradientBoostingClassifier, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "gradient boosting")
    #features_run("outcome_pos", MLPClassifier, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "mlp")
    #features_run("outcome_pos", SVC, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "support vector machine")
    #features_run("outcome_pos", tree.DecisionTreeClassifier, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "decision tree")
    #features_run("outcome_pos", SGDClassifier(loss="log", penalty="elasticnet"), UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "elastic net")   
    #features_run("outcome_neg", BaggingClassifier, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "bagging")

#max_leaf_nodes = [15]#i for i in range(2,15)]
#max_depth = [4]#i for i in range(1,5)]
#n_estimators = [10,20,30,40,50]
#learning_rate = [0.1,0.01]
#Z_data = []
#for j, c in enumerate(CLASSIFIERS): 
    
#X_data = np.array(max_leaf_nodes)
#Y_data = np.array(max_depth)
#X_data, Y_data = np.meshgrid(X_data, Y_data)
#Z_data = np.array(Z_data).reshape((len(max_depth),len(max_leaf_nodes)))

# Plot the surface.
#surface = plt.figure(1)
#ax = surface.gca(projection='3d')
#surf = ax.plot_surface(X_data, Y_data, Z_data, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)

# Add a color bar which maps values to colors.
#surface.colorbar(surf, shrink=0.5, aspect=5)
#surface.show()

#tree_plot = plt.figure(2)
#tree.plot_tree(best_model)
#tree_plot.show()
