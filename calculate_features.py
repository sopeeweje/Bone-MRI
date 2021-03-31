#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:25:26 2019

@author: Sope
"""
import csv
import os
import argparse
import glob
import nrrd
import pandas
from sklearn import preprocessing as sklearn_preprocessing
from keras.utils import to_categorical

from config import config
from filenames import ACCEPTED_FILENAMES, SKIP
from segmentation import calculate_volume

#dictionary of the various arguments for calculate_features
#--argument: type, path_to_dir, description
arguments = {'--folder': (str, config.RAW_NRRD_ROOT, 'all nrrd files folder directory'),
             '--features': (str, config.RAW_FEATURES, 'csv file of all features'),
             '--out': (str, config.FEATURES_DIR, 'output file for chosen clinical features'),
             '--nrrd': (str, config.NRRD_FEATURES, 'output file for chosen features from NRRDs in .pkl file'),
             '--pickle': (str, config.FEATURES, 'training features in .pkl file'),
             '--preprocess': (str, config.PREPROCESS, 'preprocessing features in .pkl file')
             }

#lambda functions to derive each clinical feature
clinical_feature_functions = {
    "outcome_pos": lambda f: 1 if f["category"] == "2" else 0, #positive bias = intermediate is benign
    "outcome_neg": lambda f: 1 if f["category"] == "2" or f["category"] == "1" else 0, #negative bias = intermediate is malignant
    "outcome_3": lambda f: int(f["category"]), #2-mal, 1-int, 0-ben
    "age": lambda f: float(f["age"]), #integer age
    "sex": lambda f: 1 if f["sex"] == "M" else 0, #1=male, 0=female
    "location": lambda f: get_location(f["location"]), #one-hot encoded location
    "sort": lambda f: f["sort"] #train, test, or validation
}

#functions to drive each clinical feature from the nrrd
image_feature_functions = {
    "volume": calculate_volume, #size
}

def get_location(location):
    all_locations_map = {
            "Clavicle": 0,
            "Cranium": 1,
            "Proximal femur": 2,
            "Distal femur": 3,
            "Foot": 4,
            "Proximal radius": 5,
            "Distal radius": 6,
            "Proximal ulna": 7,
            "Distal ulna": 8,
            "Hand": 9,
            "Hip": 10,
            "Proximal humerus": 11,
            "Distal humerus": 12,
            "Proximal tibia": 13,
            "Distal tibia": 14,
            "Proximal fibula": 15,
            "Distal fibula": 16,
            "Mandible": 17,
            "Rib/Chest wall": 18,
            "Scapula": 19,
            "Spine": 20,
        }
    # all_locations_map = {
    #         "Clavicle": 0,
    #         "Cranium": 1,
    #         "Femur": 2, 
    #         "Foot": 3, 
    #         "Forearm": 4,
    #         "Hand": 5,
    #         "Hip": 6,
    #         "Humerus": 7,
    #         #"Knee": 8,
    #         "Leg (Tibia/Fibula)": 8,
    #         "Mandible": 9,
    #         "Rib/Chest wall": 10,
    #         "Scapula": 11,
    #         "Spine": 12
    #     }
    all_locations_int = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    all_locations_1hot = to_categorical(all_locations_int) #make one-hot vector for each location
    this_location = all_locations_1hot[all_locations_map[location]] #pulls one-hot vector for input location
    return list(this_location)

def get_features_dict(file="/Volumes/external/bone_features.csv", id_name="patientID"):
    """
    Param:
        file - single csv with all features for all patients
    Returns:
        by_patientID - dictionary organized by patientID 
    """
    with open(file, encoding='utf-8-sig') as f:
        l = [row for row in csv.DictReader(f, skipinitialspace=True)] #{k: v for k, v in row.items()}
            #csv.DictReader(f) = convert file from csv to dictionary
            #l = list of dictionaries representing each row in the file
        by_patientID = {d[id_name]: d for d in l}
            #d for d in l - each dictionary in l
            #d[id_name] - patient_ID value
            #{d[id_name]: d for d in l} - dictionary with key=patientID, value=dictionary of all data for that patient
            #by_patientID = all_data organized by patientID
                #{patient1: {'PatientID':patient1, 'Age':50, 'Weight':200...},
                # patient2: {'PatientID':patient2, 'Age':67, 'Weight':143...},
                #...}
    return by_patientID

def all_features(files=["./features.csv"], id_name="patientID"):
    by_file = list()
    for filename in files:
        with open(filename) as f:
            l = [ {k: v for k, v in row.items() } for row in csv.DictReader(f, skipinitialspace=True )]
            by_accession = { d[id_name]: d for d in l }
            by_file.append(by_accession)
    id_sets = [ set(f.keys()) for f in by_file ]
    union = id_sets[0]
    for ids in id_sets:
        union = union | ids
    combined = dict()
    for i in union:
        c = dict()
        for by_accession in by_file:
            c = {
                **c,
                **by_accession[i],
        }
        combined[i] = c
    return combined

def get_all_nrrds(folder="/Users/Sope/Documents/GitHub/Bone-MRI/data_dir/bone_brown&china/"):
    """
    Param:
        Folder with all raw data (specifically raw NRRDs)
    Returns:
        List of all NRRDs found in that folder
    """
    all_nrrds = glob.glob("{}/**/**/**/*.nrrd".format(folder), recursive=False)
    return all_nrrds

def get_image_features(filename):
    """
    Param:
        Path to image
    Returns:
        Desired features of image in the form of a dictionary (key = feature_name, value = feature_value)
    """
    array, metadata = nrrd.read(filename)
    return {k: f(array, metadata, filename) for k, f in image_feature_functions.items()}

def get_filename_features(path):
    """
    Param:
        path to nrrd file (path = .../Source/Patient/Modality/filename.nrrd)
    Returns:
        dictionary with indicated features
    """
    split_path = path.split(os.sep)
    try:
        filename = split_path[-1]
        modality = split_path[-2]
        patientID = split_path[-3]
        features = {
            "patientID": patientID,
            "modality": modality,
            "filename": filename,
            "path": path,
        }
        return features
    except:
        return {
            "patientID": "skipped",
            "modality": "skipped",
            "filename": "skipped",
            "path": "skipped",
        }

def get_clinical_features(feat, patient):#filename):
    # patient = get_filename_features(filename)["patientID"]
    clinical = feat.get(patient, None) #all data for patient=patientID, if that pt doesn't exist return None
    if clinical is None:
        #print("missing from clinical feature sheet: {}".format(patient))
        return {}
    return {k: f(clinical) for k, f in clinical_feature_functions.items()}
        #{outcome:..., age:..., sex:...}

def filter_filenames(df):
    df = df[df.filename.isin(ACCEPTED_FILENAMES)]
    df = df[df.patientID.isin(SKIP) == 0]
    return df

def normalize_column(df, column=""):
    #min_max_scaler = sklearn_preprocessing.MinMaxScaler()
    x = df[[column]].values.astype(float)
    x_scaled = sklearn_preprocessing.scale(x) #min_max_scaler.fit_transform(x)
    x_scaled = list(zip(*list(x_scaled)))[0]
    df[column] = pandas.Series(x_scaled, index=df.index)
    return df

def features(df, filetype = 'segMask_tumor.nrrd'):
    """
    get all of the clinical features you will actually be using
    modality and filetype don't matter, just need one per patient
    """
    #df = df.drop_duplicates(["patientID","modality","filename"], 'first')
    #df = df[df.filename==filetype][["patientID", "outcome_pos", "outcome_neg", "outcome_3", "age", "sex", "location", "sort"]]#, "volume"]]
    #df = df.drop_duplicates(["patientID"], 'first')
    df = df.set_index("patientID")
    df = df.dropna()
    return df

def preprocessing(df):
    df = df.drop_duplicates(["patientID","filename","modality"], 'first')
    df = df.set_index(["patientID", "filename", "modality"]).unstack().unstack()
    return df

def get_name(n):
    return {"patientID": n}

def run(folder, features_files, out, save=True, nrrd_pickle="", features_pickle="", to_preprocess_pickle=""):
    """
    PARAM:
        folder - folder with all NRRD files
        feature_files
    """
    # nrrds = get_all_nrrds(folder) #get all nrrd files
    feat = get_features_dict(features_files) 
    patient_names = []
    with open(config.RAW_FEATURES, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[1] == "category":
                continue
            patient_names.append(row[0])
    
    # all_features = pandas.DataFrame(
    #     [{
    #         data["patientID"],
    #         data["outcome_pos"],
    #         data["age"],
    #         data["sex"],
    #         data["location"],
    #         data["sort"]
    #     } for n in data])
    # all_features = all_features.set_index("patientID")
    
    # create all features
    all_features = pandas.DataFrame(
        [{
            **get_name(n),
            #**get_filename_features(n), #patientID, modality, filename, path (4)
            **get_clinical_features(feat, n)#, #outcome_pos, outcome_neg, outcome_3, age, sex, location, sort (6)
            #**get_image_features(n) #nrrd volume (1); not confident in volume calc so omitting
        } for n in patient_names]) #nrrds])
    
    #all_features = filter_filenames(all_features)
    all_features = normalize_column(all_features, column="age")
    
    features_to_use = features(all_features) #training features
    #to_preprocess = preprocessing(all_features) #preprocessing features
    
    if save:
        features_to_use.to_csv(os.path.join(out, "training_features.csv"))
        features_to_use.to_pickle(features_pickle)
    
    # if save:
    #     all_features.to_csv(os.path.join(out, "all_features.csv"))
    #     all_features.to_pickle(nrrd_pickle)
    #     features_to_use.to_csv(os.path.join(out, "training_features.csv"))
    #     features_to_use.to_pickle(features_pickle)
    #     to_preprocess.to_csv(os.path.join(out, "preprocess.csv"))
    #     to_preprocess.to_pickle(to_preprocess_pickle)
    # else:
    #     print(to_preprocess.head())
    #     print(all_features.head())
    #     print(features_to_use.head())
    # return all_features, features_to_use, to_preprocess

if __name__ == '__main__':
    """
    Frame work of parser:
        parser.add_argument(
                '--name_of_argument', #argument specifying the location of the relevant content
                type=str, #the type to which the input is converted (i.e. string for directory)
                default=config.DEFAULT_DIRECTORY, #default location of directory if not specified
                help='description')
        FLAGS, unparsed = parser.parse_known_args()
            FLAGS = known arguments (what you've actually given values to)
            unparsed = unknown arguments (what you haven't given values to, ignored. Should be empty since there is a default)
    """
    #run("/Volumes/external/bone_raw", "/Volumes/external/features.csv", "/Volumes/external/")
    parser = argparse.ArgumentParser()
    for argument in arguments:
        parser.add_argument(
            argument,
            type=arguments[argument][0],
            default=arguments[argument][1],
            help=arguments[argument][2])
        
    parser.add_argument(
        '--temp',
        action='store_true',
        help='do not save')
    
    FLAGS, _ = parser.parse_known_args()
    run(FLAGS.folder, FLAGS.features, FLAGS.out, not FLAGS.temp, FLAGS.nrrd, FLAGS.pickle, FLAGS.preprocess)
