#git clone https://github.com/ANTsX/ANTsPy
#cd ANTsPy
#python setup.py install

import os
import logging


class Config(object):
    IMAGE_SIZE = 200

    TRIALS = 1
    BATCH_SIZE = 64
    EPOCHS = 1
    PATIENCE = 100
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    OUTCOME_BIAS = "pos"

    EXPERTS = "experts.csv"

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    RAW_NRRD_ROOT = "bone_raw/"
    RAW_FEATURES = "bone_features.csv"
    #SEQ_AVAIL = "sequence_summary.csv" #which patients have which sequences

    DATA = "data_dir/"
    PREPROCESSED_DIR = os.path.join(DATA, "preprocessed")
    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    VALIDATION_DIR = os.path.join(DATA, "validation")
    CROSSVAL_DIR = os.path.join(DATA, "crossval")
    FIGURES = "figures/"

    NOTEBOOKS = "notebooks/"

    FEATURES_DIR = "features/"
    NRRD_FEATURES = os.path.join(FEATURES_DIR, "nrrd-features.pkl")
    FEATURES = os.path.join(FEATURES_DIR, "training-features.pkl")
    PREPROCESS = os.path.join(FEATURES_DIR, "preprocess.pkl")

    INPUT_FORM = "t2"

    OUTPUT = "output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")

    MAIN_TEST_HOLDOUT = 0.2
    NUMBER_OF_FOLDS = 4
    SPLIT_TRAINING_INTO_VALIDATION = 0.1


config = Config()
