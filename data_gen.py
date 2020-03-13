import uuid
import traceback
import os
import numpy as np
import pandas
import nrrd
import glob
import argparse
import random
from PIL import Image
import csv
from shutil import rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, Iterator
from tqdm import tqdm

from segmentation import calculate_percentile_slice, select_slice, bounding_box, crop, resize, calculate_volume
from config import config

from filenames import IMAGE, SEGMENTATION, T1, T2, T1C

clinical_features = [
    #"age",
    "volume",
    #"sex"
]

def all_input(t1, t2, t1c, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    t1c_image = np.array(t1c)
    t1c_image = np.rollaxis(t1c_image, 0, 3)
    return (t1_image, t2_image, t1c_image), features, labels

def t1_input(t1, t2, t1c, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None, None), [], labels

def t1c_input(t1, t2, t1c, features, labels):
    t1c_image = np.array(t1c)
    t1c_image = np.rollaxis(t1c_image, 0, 3)
    return (None, None, t1c_image), [], labels

def t2_input(t1, t2, t1c, features, labels):
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (None, t2_image, None), [], labels

def t1_t2_input(t1, t2, t1c, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (t1_image, t2_image, None), [], labels

def t1c_t2_input(t1, t2, t1c, features, labels):
    t1c_image = np.array(t1c)
    t1c_image = np.rollaxis(t1c_image, 0, 3)
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (None, t2_image, t1c_image), [], labels

def t1_features_input(t1, t2, t1c, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None, None), features, labels

def t2_features_input(t1, t2, t1c, features, labels):
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (None, t2_image, None), features, labels

def features_input(t1, t2, t1c, features, labels):
    return (None, None, None), features, labels

def t1c_features_input(t1, t2, t1c, features, labels):
    t1c_image = np.array(t1c)
    t1c_image = np.rollaxis(t1c_image, 0, 3)
    return (None, None, t1c_image), features, labels

INPUT_FORMS = {
    "all": all_input,
    "t1": t1_input,
    "t2": t2_input,
    "t1c": t1c_input,
    "t1-t2": t1_t2_input,
    "t1c-t2": t1c_t2_input,
    "t1-features": t1_features_input,
    "t2-features": t2_features_input,
    "t1c-features": t1c_features_input,
    "features": features_input,
}

INPUT_FORM_PARAMETERS = {
    "all": {
        "t1": True,
        "t2": True,
        "t1c": True,
        "features": True,
    },
    "t1": {
        "t1": True,
        "t2": False,
        "t1c": False,
        "features": False,
    },
    "t2": {
        "t1": False,
        "t2": True,
        "t1c": False,
        "features": False,
    },
    "t1c": {
        "t1": False,
        "t2": False,
        "t1c": True,
        "features": False,
    },
    "t1-t2": {
        "t1": True,
        "t2": True,
        "t1c": False,
        "features": False,
    },
    "t1c-t2": {
        "t1": False,
        "t2": True,
        "t1c": True,
        "features": False,
    },
    "t1-features": {
        "t1": True,
        "t2": False,
        "t1c": False,
        "features": True,
    },
    "t2-features": {
        "t1": False,
        "t2": True,
        "t1c": False,
        "features": True,
    },
    "t1c-features": {
        "t1": False,
        "t2": False,
        "t1c": True,
        "features": True,
    },
    "features": {
        "t1": False,
        "t2": False,
        "t1c": False,
        "features": True,
    },
}
    
available = {}
with open(config.SEQ_AVAIL) as seq_avail:
   reader = csv.reader(seq_avail)
   headers = next(reader, None)
   for h in headers:
       available[h] = []
   for row in reader:
       available['pd'].append(row[0])
       available['t1'].append(row[1])
       available['t2'].append(row[2])
       available['t1c'].append(row[3])

class Features(Iterator):
    def __init__(self, features, shuffle, seed):
        super(Features, self).__init__(len(features), config.BATCH_SIZE, shuffle, hash(seed) % 2**32 )
        self.features = np.array(features)

    def _get_batches_of_transformed_samples(self, index_array):
        return self.features[index_array]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

class Dataset(object):
    def __init__(self, images, features, labels, names, augment=False, shuffle=False, seed=None, input_form="all"):
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment
        self.input_form = input_form
        self.names = names

        self.parameters = INPUT_FORM_PARAMETERS[input_form]

        features = list(zip(*features))

        self.labels = labels

        self.features = features
        self.features_size = 0
        if self.parameters["features"]:
            self.features_size = len(features[0])
            self.features_generator = Features(self.features, self.shuffle, self.seed)

        self.n = len(labels)

        unique, index, inverse, counts = np.unique(self.labels, return_index=True, return_inverse=True, return_counts=True)
        self.y = inverse
        self.classes = inverse
        self.class_indices = { u: i for i, u in enumerate(unique) }

        separate_images = list(zip(*images))
        if self.parameters["t1"]:
            self.t1 = np.array(separate_images[0])
            self.datagen1 = self._get_data_generator()
        if self.parameters["t2"]:
            self.t2 = np.array(separate_images[1])
            self.datagen2 = self._get_data_generator()
        if self.parameters["t1c"]:
            self.t1c = np.array(separate_images[2])
            self.datagen1c = self._get_data_generator()

        self.reset()

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        if self.parameters["features"]:
            self.features_generator = Features(self.features, self.shuffle, self.seed)

        if self.parameters["t1"]:
            self.generator_t1 = self.datagen1.flow(
                    x=self.t1,
                    y=self.y,
                    batch_size=config.BATCH_SIZE,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32,
                    )

        if self.parameters["t2"]:
            self.generator_t2 = self.datagen2.flow(
                    x=self.t2,
                    y=self.y,
                    batch_size=config.BATCH_SIZE,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32 ,
                    )
            
        if self.parameters["t1c"]:
            self.generator_t1c = self.datagen1c.flow(
                    x=self.t1c,
                    y=self.y,
                    batch_size=config.BATCH_SIZE,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32 ,
                    )
            
        self.labels_generator = Features(self.y, self.shuffle, self.seed)

    def _get_data_generator(self):
        if self.augment:
            return ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
            )
        return ImageDataGenerator(
            rescale=1. / 255,
            )

    def next(self):
        labels = self.labels_generator.next()
        inputs = list()
        if self.parameters["t2"]:
            inputs.append(self.generator_t2.next()[0])
        if self.parameters["t1"]:
            inputs.append(self.generator_t1.next()[0])
        if self.parameters["t1c"]:
            inputs.append(self.generator_t1c.next()[0])
        if self.parameters["features"]:
            inputs.append(self.features_generator.next())
        if len(inputs) == 1:
            inputs = inputs[0]
        return (inputs, labels)
    
    def get_names(self):
        return self.names

def outcome_feature(row):
    label = row["outcome"]
    features = [ row[f] for f in clinical_features ]
    return label, features

LABEL_FORMS = {
    "outcome": outcome_feature,
}

def get_label_features(row, label="outcome"):
    """returns label, features, sample name"""
    return (*LABEL_FORMS[label](row), row.name)

def input_data_form(t1, t2, t1c, features, labels, input_form=config.INPUT_FORM):
    images, features, labels = INPUT_FORMS[input_form](t1, t2, t1c, features, labels)
    return images, features, labels

def load_image(image_path, segmentation_path, verbose=False):
    image, _ = nrrd.read(image_path)
    segmentation, _ = nrrd.read(segmentation_path)
    if verbose:
        print("""
        image: {}
        seg: {}
""".format(image.shape, segmentation.shape))
    return [mask_image_percentile(image, segmentation, 100, a) for a in (0, 1, 2)]

def mask_image_percentile(image, segmentation, percentile=100, axis=2):
    plane = calculate_percentile_slice(segmentation, percentile, axis)
    image, segmentation = select_slice(image, segmentation, plane, axis)

    bounds = bounding_box(segmentation)
    image, segmentation = crop(image, segmentation, bounds)

    masked = image * segmentation
    masked = resize(masked, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return masked


SHAPES_OUTPUT = """
SHAPES
    {}:"""

def generate_from_features(df, input_form=config.INPUT_FORM, label_form="outcome", verbose=False, source=config.PREPROCESSED_DIR):
    parameters = INPUT_FORM_PARAMETERS[input_form]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        print(index)
        t1_image_file = os.path.join(source, "{}-{}-{}".format(index, T1, IMAGE))
        t1_seg_file = os.path.join(source, "{}-{}-{}".format(index, T1, SEGMENTATION))
        t2_image_file = os.path.join(source, "{}-{}-{}".format(index, T2, IMAGE))
        t2_seg_file = os.path.join(source, "{}-{}-{}".format(index, T2, SEGMENTATION))
        t1c_image_file = os.path.join(source, "{}-{}-{}".format(index, T1C, IMAGE))
        t1c_seg_file = os.path.join(source, "{}-{}-{}".format(index, T1C, SEGMENTATION))
        
        t1_masked = None
        t2_masked = None
        t1c_masked = None
        
        try:
            if (parameters["t1"] and index in available['t1']) or parameters["features"]: # load in case of features so that files that error out aren't included in analysis
                if verbose:
                    print(SHAPES_OUTPUT.format("t1"))
                t1_masked = load_image(t1_image_file, t1_seg_file, verbose=verbose)
        except Exception as e:
            print()
            print("#" * 80)
            print("Exception occurred for: {}\n{}".format(row, e))
            print("T1 image unavailable")
            print(traceback.format_exc())
            continue
            
        try:
            if parameters["t2"] and index in available['t2']:
                if verbose:
                    print(SHAPES_OUTPUT.format("t2"))
                t2_masked = load_image(t2_image_file, t2_seg_file, verbose=verbose)
        except Exception as e:
            print()
            print("#" * 80)
            print("Exception occurred for: {}\n{}".format(row, e))
            print("T2 image unavailable")
            print(traceback.format_exc())
            continue
        
        try:
            if parameters["t1c"] and index in available['t1c']: 
                if verbose:
                    print(SHAPES_OUTPUT.format("t1c"))
                t1c_masked = load_image(t1c_image_file, t1c_seg_file, verbose=verbose)
        except Exception as e:
            print()
            print("#" * 80)
            print("Exception occurred for: {}\n{}".format(row, e))
            print("T1C image unavailable")
            print(traceback.format_exc())
            continue
            
        labels, features, name = get_label_features(row, label=label_form)
        images, features, labels = input_data_form(t1_masked, t2_masked, t1c_masked, features, labels, input_form=input_form)
        yield images, features, labels, name

def sort(validation_fraction=0.2, test_fraction=0.1, seed=None, label_form="outcome", input_form="all"):
    f = pandas.read_pickle(config.FEATURES) #CSV with all training features for all available patients
    train_fraction = 1 - validation_fraction - test_fraction
    input_form_map = {
        "all": lambda f: f[f.index.isin(available['t1']).isin(available['t2']).isin(available['t1c'])],
        "t1": lambda f: f[f.index.isin(available['t1'])],
        "t2": lambda f: f[f.index.isin(available['t2'])],
        "t1c": lambda f: f[f.index.isin(available['t1c'])],
        "t1-t2": lambda f: f[f.index.isin(available['t1']) & f.index.isin(available['t2'])],
        "t1c-t2": lambda f: f[f.index.isin(available['t1c']) & f.index.isin(available['t2'])],
        "t1-features": lambda f: f[f.index.isin(available['t1'])],
        "t2-features": lambda f: f[f.index.isin(available['t2'])],
        "t1c-features": lambda f: f[f.index.isin(available['t1c'])],
        "features": lambda f: f
        }

    f = input_form_map[input_form](f) #CSV with all training features for patients with given imaging modality
    remaining = f.copy()

    sort_dict = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }

    # calculate goal numbers for train/validation/test by label properties
    labels = f[label_form].unique() # 1 (malignant) or 0 (benign)
    goal_sort = dict()
    for l in labels:
        label_fraction = len(remaining[remaining[label_form] == l])/len(remaining) # no. pts. with given label/total no. of pts. = % benign (ex.)
        for s in ["train", "validation", "test"]:
            goal_sort[(l, s)] = int(len(remaining) * label_fraction * sort_dict[s]) #ex. goal_sort[(benign, train)] = total no. of pts. * %benign * %train

    all_train = list()
    all_validation = list()
    all_test = list()
    sorted_dict = {
        "train": all_train,
        "validation": all_validation,
        "test": all_test,
    }

    # get preassigned sorts
    train = f[f["sort"] == "train"] #all patients pre-assigned to training dataset
    validation = f[f["sort"] == "validation"]
    test = f[f["sort"] == "test"]
    presort_dict = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    # recalculate goals based on preassigned sorts
    for s in ["train", "validation", "test"]:
        presorted = presort_dict[s]
        for l in labels:
            goal_sort[(l, s)] = max(0, goal_sort[(l, s)] - len(presorted[presorted[label_form] == l]))
    # add preassigned sorts and remove from lesions to sort
    all_train.append(train)
    all_validation.append(validation)
    all_test.append(test)
    remaining = remaining.drop(train.index)
    remaining = remaining.drop(validation.index)
    remaining = remaining.drop(test.index)

    # sort remaining lesions
    for l in labels:
        for s in ["train", "validation", "test"]:
            label_set = remaining[remaining[label_form] == l]
            label_set = label_set.sample(n = min(goal_sort[(l, s)], len(label_set)), random_state=(int(seed) % 2 ** 32))
            remaining = remaining.drop(label_set.index)
            sorted_dict[s].append(label_set)
    # append any left over
    all_train.append(remaining)

    train = pandas.concat(all_train)
    validation = pandas.concat(all_validation)
    test = pandas.concat(all_test)

#    print("Training points = " + str(len(train)))
#    print("Benign: " + str(len(train[train[label_form] == 0])) + ", Malignant: " + str(len(train[train[label_form] == 1])))
#    print("")
#    print("Validation points = " + str(len(validation)))
#    print("Benign: " + str(len(validation[validation[label_form] == 0])) + ", Malignant: " + str(len(validation[validation[label_form] == 1])))
#    print("")
#    print("Testing points = " + str(len(test)))
#    print("Benign: " + str(len(test[test[label_form] == 0])) + ", Malignant: " + str(len(test[test[label_form] == 1])))

    train.to_csv(os.path.join(config.TRAIN_DIR, "{}-train.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.VALIDATION_DIR, "{}-validation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.TEST_DIR, "{}-test.csv".format(str(seed))))

    return train, validation, test

def relist(l):
    l = list(l)
    if len(l) == 0:
        return l
    return [[k[i] for k in l] for i, _ in enumerate(l[0])]

def data(seed=None,
        input_form=config.INPUT_FORM,
        label_form="outcome",
        train_shuffle=True,
        validation_shuffle=False,
        test_shuffle=False,
        train_augment=True,
        validation_augment=False,
        test_augment=False,
        validation_split=config.VALIDATION_SPLIT,
        test_split=config.TEST_SPLIT,
        verbose=False,
        ):
    train, validation, test = sort(validation_split, test_split, seed, label_form, input_form)
    train_images, train_features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))
    validation_images, validation_features, validation_labels, validation_names = relist(generate_from_features(validation, input_form=input_form, label_form=label_form, verbose=verbose))
    test_images, test_features, test_labels, test_names = relist(generate_from_features(test, input_form=input_form, label_form=label_form, verbose=verbose))

    train_features = relist(train_features)
    validation_features = relist(validation_features)
    test_features = relist(test_features)

    train_generator = Dataset(
            train_images,
            train_features,
            train_labels,
            train_names,
            augment=train_augment,
            shuffle=train_shuffle,
            input_form=input_form,
            seed=seed,
        )
    validation_generator = Dataset(
            validation_images,
            validation_features,
            validation_labels,
            validation_names,
            augment=validation_augment,
            shuffle=validation_shuffle,
            input_form=input_form,
            seed=seed,
        )
    test_generator = Dataset(
            test_images,
            test_features,
            test_labels,
            test_names,
            augment=test_augment,
            shuffle=test_shuffle,
            input_form=input_form,
            seed=seed,
        )
    return train_generator, validation_generator, test_generator

def load_from_features(
        features,
        input_form=config.INPUT_FORM,
        label_form="outcome",
        source=config.PREPROCESSED_DIR,
        shuffle=True,
        augment=True,
        verbose=False,
        ):
    images, features, labels, names = relist(generate_from_features(features, input_form=input_form, label_form=label_form, verbose=verbose, source=source))
    features = relist(features)

    generator = Dataset(
            images,
            features,
            labels,
            names,
            augment=augment,
            shuffle=shuffle,
            input_form=input_form,
            seed=0,
        )
    return generator

def xdata(fold_number,
          train,
          validation,
          test,
          # holdout_test,
          seed=None,
          input_form=config.INPUT_FORM,
          label_form="outcome",
          train_shuffle=True,
          validation_shuffle=False,
          test_shuffle=False,
          train_augment=True,
          validation_augment=False,
          test_augment=False,
          verbose=True
          ):

    #save the data in each set for the fold run
    fold_string = 'fold-' + str(fold_number)
    train.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-ktrain.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-kvalidation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-ktest.csv".format(str(seed))))
    #holdout_test.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-kholdouttest.csv".format(str(seed))))

    # loading of the features - this is supposed to be the bottleneck, but seems to be pretty fast when I was testing it; refactor later
    train_images, train_features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))
    validation_images, validation_features, validation_labels, validation_names = relist(generate_from_features(validation, input_form=input_form, label_form=label_form, verbose=verbose))
    test_images, test_features, test_labels, test_names = relist(generate_from_features(test, input_form=input_form, label_form=label_form, verbose=verbose))
    # holdouttest_images, holdouttest_features, holdouttest_labels, holdouttest_names = relist(generate_from_features(holdout_test, input_form=input_form, label_form=label_form, verbose=verbose))

    train_features = relist(train_features)
    validation_features = relist(validation_features)
    test_features = relist(test_features)
    # holdouttest_features = relist(holdouttest_features)

    train_generator = Dataset(
            train_images,
            train_features,
            train_labels,
            train_names,
            augment=train_augment,
            shuffle=train_shuffle,
            input_form=input_form,
            seed=seed,
        )
    validation_generator = Dataset(
            validation_images,
            validation_features,
            validation_labels,
            validation_names,
            augment=validation_augment,
            shuffle=validation_shuffle,
            input_form=input_form,
            seed=seed,
        )
    test_generator = Dataset(
            test_images,
            test_features,
            test_labels,
            test_names,
            augment=test_augment,
            shuffle=test_shuffle,
            input_form=input_form,
            seed=seed,
        )
    # holdout_test_generator = Dataset(holdouttest_images,holdouttest_features,holdouttest_labels,holdouttest_names,augment=test_augment,shuffle=test_shuffle,input_form=input_form,seed=seed,)
    return train_generator, validation_generator, test_generator  # , holdout_test_generator

if __name__ == '__main__':
    data(uuid.uuid4())
