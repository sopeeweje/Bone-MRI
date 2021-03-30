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
from keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt

from segmentation import calculate_percentile_slice, select_slice, bounding_box, crop, resize, calculate_volume, calculate_top3_slices
from config import config

from filenames import IMAGE, SEGMENTATION, T1, T2, T1C

clinical_features = [
    "age",
    "sex",
    "location"
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
    
#available = {}
#with open(config.SEQ_AVAIL) as seq_avail:
#   reader = csv.reader(seq_avail)
#   headers = next(reader, None)
#   for h in headers:
#       available[h] = []
#   for row in reader:
#       available['pd'].append(row[1])
#       available['t1'].append(row[2])
#       available['t1c'].append(row[3])
#       available['t2'].append(row[4])
#       available['t2-t1'].append(row[5])
#       available['t2-t1c'].append(row[6])

class Features(Iterator):
    """
    tf.keras.preprocessing.image.Iterator(n, batch_size, shuffle, seed)
    
    n: Integer, total number of samples in the dataset to loop over.
    batch_size: Integer, size of a batch.
    shuffle: Boolean, whether to shuffle the data between epochs.
    seed: Random seeding for data shuffling.
    """
    def __init__(self, features, shuffle, seed):
        super(Features, self).__init__(len(features), config.BATCH_SIZE, shuffle, hash(seed) % 2**32 )
        self.features = np.array(features)
            
    def _get_batches_of_transformed_samples(self, index_array):
        return self.features[index_array] #get features of given batch

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

class Dataset(object):
    def __init__(self, images, features, labels, names, augment=False, shuffle=False, seed=None, input_form="all", three=False):
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment
        self.input_form = input_form
        self.names = names

        self.parameters = INPUT_FORM_PARAMETERS[input_form] #ex. {"t1": True, "t2": False, "t1c": False, "features": False}

        features = list(zip(*features))

        self.labels = labels

        self.features = features
        self.features_size = 0
        self.features_int = []
        self.features_list = []
        for row in self.features: #ex. [0,1,[0,0,0,1,0,0,0,0,0,0,0,0,0,0]]
            new_row_int = []
            new_row_list = []
            for item in row:
                if not isinstance(item, list):
                    new_row_int.append(item) # [0,43]
                else: #if it's a list
                    new_row_list = item #[0,0,0,1,0,0,0,0,0,0,0,0,0,0]
            self.features_int.append(new_row_int) #[[0,43], [1,12],...]
            self.features_list.append(new_row_list) #[[0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0],...]
        if self.parameters["features"]:
            self.features_size = len(features[0]) #3 -> [age, sex, location]
            self.features_int_generator = Features(self.features_int, self.shuffle, self.seed)
            self.features_list_generator = Features(self.features_list, self.shuffle, self.seed)
            self.features_generator = Features(self.features, self.shuffle, self.seed)

        self.n = len(labels)

        unique, index, inverse, counts = np.unique(self.labels, return_index=True, return_inverse=True, return_counts=True)
        self.y = inverse
        if three:
            self.y = to_categorical(inverse)
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
            self.features_int_generator = Features(self.features_int, self.shuffle, self.seed)
            self.features_list_generator = Features(self.features_list, self.shuffle, self.seed)
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
                    seed=hash(self.seed) % 2**32,
                    )
            
        if self.parameters["t1c"]:
            self.generator_t1c = self.datagen1c.flow(
                    x=self.t1c,
                    y=self.y,
                    batch_size=config.BATCH_SIZE,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32,
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
            inputs.append([self.features_int_generator.next(),self.features_list_generator.next()])
            #inputs.append(self.features_generator.next())
        if len(inputs) == 1:
            inputs = inputs[0]
        return (inputs, labels)
    
    def get_names(self):
        return self.names

def outcome_feature(row, label):
    label = row[label]
    features = [ row[f] for f in clinical_features ]
    return label, features #(label, [age, sex, location]) -> ex. (1, [32, 0, [0,0,0,0,1,0,0,0,0,0,0,0,0,0]])

LABEL_FORMS = {
    "outcome_3": outcome_feature,
    "outcome_pos": outcome_feature,
    "outcome_neg": outcome_feature
}

def get_label_features(row, label="outcome"):
    """returns label, features, sample name"""
    return (*LABEL_FORMS[label](row, label), row.name) #(outcome_feature(row, label), row.name) -> (label, features, patientID)

def input_data_form(t1, t2, t1c, features, labels, input_form=config.INPUT_FORM):
    images, features, labels = INPUT_FORMS[input_form](t1, t2, t1c, features, labels) #ex. t2_input(t1, t2, t1c, features, labels)
    #ex. for t2 input -> (None, t2_image, None), [], labels
    #ex. for features input -> (None, None, None), features, labels
    return images, features, labels 

def load_image(image_path, segmentation_path, verbose=False):
    image, _ = nrrd.read(image_path) #import image nrrd
    segmentation, _ = nrrd.read(segmentation_path) #import segmentation nrrd
    if verbose:
        print("""
        image: {}
        seg: {}
""".format(image.shape, segmentation.shape))
    return [mask_image_percentile(image, segmentation, 100, a) for a in (0, 1, 2)] #3 masked images, 1 for each axis. changed to use the "actual" image on each channel, cuts lack appropriate resolution
    #return multicut_1axis_mask(image, segmentation, axis=2)

def mask_image_percentile(image, segmentation, percentile=100, axis=2):
    if image.ndim == 4:
        print("4dim")
        image = image[0]

    plane = calculate_percentile_slice(segmentation, percentile, axis) #find largest slice
    image, segmentation = select_slice(image, segmentation, plane, axis) #select the largest slide

    bounds = bounding_box(segmentation) #create a bounding box the size of the segmentation
    image, segmentation = crop(image, segmentation, bounds) #crop to size of the segmentation

    masked = image * segmentation #mask everything in image except what was segmented
    masked = resize(masked, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #resize to desired image size
    #plt.imsave("data_gen_images/" + str(uuid.uuid4()) + ".png", masked)
    #print(np.isnan(masked).any())
    return masked

def multicut_1axis_mask(image, segmentation, axis=2):
    plane1, plane2, plane3 = calculate_top3_slices(segmentation, axis) #find largest slice
    if image.ndim == 4:
        print("4dim")
        image = image[0]
    
    image1, segmentation1 = select_slice(image, segmentation, plane1, axis) #select the largest slice
    bounds = bounding_box(segmentation1) #create a bounding box the size of the segmentation
    image1, segmentation1 = crop(image1, segmentation1, bounds) #crop to size of the segmentation
    masked1 = image1 * segmentation1 #mask everything in image except what was segmented
    masked1 = resize(masked1, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #resize to desired image size
    
    image2, segmentation2 = select_slice(image, segmentation, plane2, axis) #select the second largest slice
    bounds = bounding_box(segmentation2) #create a bounding box the size of the segmentation
    image2, segmentation2 = crop(image2, segmentation2, bounds) #crop to size of the segmentation
    masked2 = image2 * segmentation2 #mask everything in image except what was segmented
    masked2 = resize(masked2, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #resize to desired image size
    
    image3, segmentation3 = select_slice(image, segmentation, plane3, axis) #select the third largest slice
    bounds = bounding_box(segmentation3) #create a bounding box the size of the segmentation
    image3, segmentation3 = crop(image3, segmentation3, bounds) #crop to size of the segmentation
    masked3 = image3 * segmentation3 #mask everything in image except what was segmented
    masked3 = resize(masked3, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #resize to desired image size
    #plt.imsave("/Volumes/external/datagentest.png", masked1)
    
    return [masked1, masked2, masked3]
    
SHAPES_OUTPUT = """
SHAPES
    {}:"""

def generate_from_features(df, input_form=config.INPUT_FORM, label_form="outcome", verbose=False, source=config.PREPROCESSED_DIR):
    #determine which features you need to get
    #ex. {"t1": True, "t2": False, "t1c": False, "features": False}
    parameters = INPUT_FORM_PARAMETERS[input_form]

    for index, row in tqdm(df.iterrows(), total=len(df)): #for each patient
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
        
        if parameters["t1"]:
            try:
                print("I'm doing the thing.")
                if verbose:
                    print(SHAPES_OUTPUT.format("t1"))
                t1_masked = load_image(t1_image_file, t1_seg_file, verbose=verbose)
                #print("I'm doing the thing.")
                #if verbose:
                #    print(SHAPES_OUTPUT.format("t1c"))
                #t1c_masked = load_image(t1c_image_file, t1c_seg_file, verbose=verbose)
            except Exception as e:
                print()
                print("#" * 80)
                print("Exception occurred for: {}\n{}".format(row, e))
                print("T1 image unavailable")
                print(traceback.format_exc())
                continue
        
        if parameters["t2"]:
            try:
                print("I'm doing the thing.")
                if verbose:
                    print(SHAPES_OUTPUT.format("t2"))
                t2_masked = load_image(t2_image_file, t2_seg_file, verbose=verbose)
                print("I'm doing the thing.")
                #if verbose:
                #    print(SHAPES_OUTPUT.format("t1c"))
                #t1c_masked = load_image(t1c_image_file, t1c_seg_file, verbose=verbose)
            except Exception as e:
                print()
                print("#" * 80)
                print("Exception occurred for: {}\n{}".format(row, e))
                print("T2 image unavailable")
                print(traceback.format_exc())
                continue

        if parameters["t1c"]:
            try:
                print("I'm doing the thing.")
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
        
        #if parameters["features"]:
        #    try:
        #        print("I'm doing the thing.")
        #        if verbose:
        #            print(SHAPES_OUTPUT.format("t1c"))
        #        t1c_masked = load_image(t1c_image_file, t1c_seg_file, verbose=verbose)
        #    except Exception as e:
        #        print()
        #        print("#" * 80)
        #        print("Exception occurred for: {}\n{}".format(row, e))
        #        print("T1C image unavailable")
        #        print(traceback.format_exc())
        #        continue
        #ex. (1, [32, 0, [0,0,0,0,1,0,0,0,0,0,0,0,0,0]], "bone-penn-453")
        labels, features, name = get_label_features(row, label=label_form)

        #ex. for t2 input -> (None, t2_image, None), [], labels
        #ex. for features input -> (None, None, None), features, labels
        images, features, labels = input_data_form(t1_masked, t2_masked, t1c_masked, features, labels, input_form=input_form)
        
        #ex. (None, t2_image, None), [], labels, "bone-penn-453"
        yield images, features, labels, name

def sort(validation_fraction=0.2, test_fraction=0.1, seed=None, label_form="outcome", input_form="all"):
    f = pandas.read_pickle(config.FEATURES) #pickle with all training features for all available patients
    print(f)
    print(len(f))
    f = f.drop('bone-penn-479')
    print(len(f))
    train_fraction = 1 - validation_fraction - test_fraction #fraction of points going to training set
    
    #filter data set to include only patients with desired studies available
    #input_form_map = {
    #    "all": lambda f: f[f.index.isin(available['t1']).isin(available['t2']).isin(available['t1c'])],
    #    "t1": lambda f: f[f.index.isin(available['t2-t1'])], #only using patients with T1 and T2
    #    "t2": lambda f: f[f.index.isin(available['t2-t1'])], #only using patients with T1 and T2
    #    "t1c": lambda f: f[f.index.isin(available['t1c'])],
    #    "t2-t1": lambda f: f[f.index.isin(available['t2-t1'])],
    #    "t1c-t2": lambda f: f[f.index.isin(available['t1c']) & f.index.isin(available['t2'])],
    #    "t1-features": lambda f: f[f.index.isin(available['t1'])],
    #    "t2-features": lambda f: f[f.index.isin(available['t2'])],
    #    "t1c-features": lambda f: f[f.index.isin(available['t1c'])],
    #    "features": lambda f: f
    #    }
    
    #f = input_form_map[input_form](f) #CSV with all training features for patients with given imaging modality
    remaining = f.copy()

    sort_dict = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    
    # external sort
    all_external = list()
    external = f[f["sort"] == "external"] #all patients pre-assigned to training 
    all_external.append(external)
    remaining = remaining.drop(external.index)

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
    train = f[f["sort"] == "train"] #all patients pre-assigned to training 
    validation = f[f["sort"] == "validation"] #all patients pre-assigned to validation 
    test = f[f["sort"] == "test"] #all patients pre-assigned to testing 
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
    
    # append any left over to training set
    all_train.append(remaining)
    
    train = pandas.concat(all_train)
    validation = pandas.concat(all_validation)
    test = pandas.concat(all_test)
    external = pandas.concat(all_external)

    print("Training points = " + str(len(train)))
    print("Benign: " + str(len(train[train[label_form] == 0])) + ", Malignant: " + str(len(train[train[label_form] == 1])))
    print("")
    print("Validation points = " + str(len(validation)))
    print("Benign: " + str(len(validation[validation[label_form] == 0])) + ", Malignant: " + str(len(validation[validation[label_form] == 1])))
    print("")
    print("Testing points = " + str(len(test)))
    print("Benign: " + str(len(test[test[label_form] == 0])) + ", Malignant: " + str(len(test[test[label_form] == 1])))
    print("External points = " + str(len(external)))
    print("Benign: " + str(len(external[external[label_form] == 0])) + ", Malignant: " + str(len(external[external[label_form] == 1])))
    
    #CSV format:
    #[[patientID	, outcome_pos, outcome_neg, outcome_3, age, sex, location, sort]
    # [bone-penn-547, 0, 1, 1, 12, 0, [0,0,1,0,0,0,0,0,0,0,0,0,0,0], -],
    #...]

    train.to_csv(os.path.join(config.TRAIN_DIR, "{}-train.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.VALIDATION_DIR, "{}-validation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.TEST_DIR, "{}-test.csv".format(str(seed))))
    external.to_csv(os.path.join(config.EXTERNAL_DIR, "{}-external.csv".format(str(seed))))
    return train, validation, test, external #return dataframes

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
    
    #sort which points go into which dataset
    train, validation, test, external = sort(validation_split, test_split, seed, label_form, input_form)
    
    #generate images, features, labels, and name (patientIDs). relist makes list of yielded values
    train_images, train_features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))
    validation_images, validation_features, validation_labels, validation_names = relist(generate_from_features(validation, input_form=input_form, label_form=label_form, verbose=verbose))
    test_images, test_features, test_labels, test_names = relist(generate_from_features(test, input_form=input_form, label_form=label_form, verbose=verbose))
    external_images, external_features, external_labels, external_names = relist(generate_from_features(external, input_form=input_form, label_form=label_form, verbose=verbose))
    
    train_features = relist(train_features)
    validation_features = relist(validation_features)
    test_features = relist(test_features)
    external_features = relist(external_features)

    print("Training: {}".format(len(train_labels)))
    print("Validation: {}".format(len(validation_labels)))
    print("Testing: {}".format(len(test_labels)))
    print("External: {}".format(len(external_labels)))

    three = (label_form == "outcome_3")
    train_generator = Dataset(
            train_images,
            train_features,
            train_labels,
            train_names,
            augment=train_augment,
            shuffle=train_shuffle,
            input_form=input_form,
            seed=seed,
            three=three
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
            three=three,
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
            three=three
        )
    external_generator = Dataset(
            external_images,
            external_features,
            external_labels,
            external_names,
            augment=test_augment,
            shuffle=test_shuffle,
            input_form=input_form,
            seed=seed,
            three=three
        )
    return train_generator, validation_generator, test_generator, external_generator

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
          holdout_test,
          seed=None,
          input_form=config.INPUT_FORM,
          label_form="outcome",
          train_shuffle=True,
          validation_shuffle=False,
          test_shuffle=False,
          train_augment=True,
          validation_augment=False,
          test_augment=False,
          verbose=False
          ):

    #save the data in each set for the fold run
    fold_string = 'fold-' + str(fold_number)
    train.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-ktrain.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-kvalidation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-ktest.csv".format(str(seed))))
    holdout_test.to_csv(os.path.join(config.CROSSVAL_DIR, fold_string + "-{}-kholdouttest.csv".format(str(seed))))

    # loading of the features - this is supposed to be the bottleneck, but seems to be pretty fast when I was testing it; refactor later
    train_images, train_features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))
    validation_images, validation_features, validation_labels, validation_names = relist(generate_from_features(validation, input_form=input_form, label_form=label_form, verbose=verbose))
    test_images, test_features, test_labels, test_names = relist(generate_from_features(test, input_form=input_form, label_form=label_form, verbose=verbose))
    holdouttest_images, holdouttest_features, holdouttest_labels, holdouttest_names = relist(generate_from_features(holdout_test, input_form=input_form, label_form=label_form, verbose=verbose))

    train_features = relist(train_features)
    validation_features = relist(validation_features)
    test_features = relist(test_features)
    holdouttest_features = relist(holdouttest_features)

    three = (label_form == "outcome_3")

    train_generator = Dataset(
            train_images,
            train_features,
            train_labels,
            train_names,
            augment=train_augment,
            shuffle=train_shuffle,
            input_form=input_form,
            seed=seed,
            three=three
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
            three=three
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
            three=three
        )
    holdout_test_generator = Dataset(holdouttest_images,holdouttest_features,holdouttest_labels,holdouttest_names,augment=test_augment,shuffle=test_shuffle,input_form=input_form,seed=seed,)
    return train_generator, validation_generator, test_generator, holdout_test_generator

#if __name__ == '__main__':
    #data(uuid.uuid4())
