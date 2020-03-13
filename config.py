#git clone https://github.com/ANTsX/ANTsPy
#cd ANTsPy
#python setup.py install

import os
import logging


class Config(object):
    IMAGE_SIZE = 200

    TRIALS = 10
    BATCH_SIZE = 16

    EPOCHS = 500
    PATIENCE = 100
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    OUTCOME_BIAS = "pos"

    EXPERTS = "/Users/feyisope/Documents/Radiology-ML/test/Ovarian_Experts.csv" #check

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    CHOP_IDS = '/home/sopeeweje/bone_master/chopIDs.csv'

    RAW_NRRD_ROOT = "/home/sopeeweje/bone_master/bone_raw"
    RAW_FEATURES = "/home/sopeeweje/bone_master/bone_features.csv"
    SEQ_AVAIL = "/home/sopeeweje/bone_master/sequence_summary.csv" #which patients have which sequences

    DATA = "/home/sopeeweje/bone_master/data_dir"
    PREPROCESSED_DIR = os.path.join(DATA, "preprocessed")
    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    VALIDATION_DIR = os.path.join(DATA, "validation")
    CROSSVAL_DIR = os.path.join(DATA, "crossval")

    FEATURES_DIR = "/home/sopeeweje/bone_master/features"
    NRRD_FEATURES = os.path.join(FEATURES_DIR, "nrrd-features.pkl")
    FEATURES = os.path.join(FEATURES_DIR, "training-features.pkl")
    PREPROCESS = os.path.join(FEATURES_DIR, "preprocess.pkl")

    INPUT_FORM = "t1c-features"

    OUTPUT = "/home/sopeeweje/bone_master/output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")

    MAIN_TEST_HOLDOUT = 0.2
    NUMBER_OF_FOLDS = 4
    SPLIT_TRAINING_INTO_VALIDATION = 0.1


config = Config()
