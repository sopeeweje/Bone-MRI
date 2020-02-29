import csv
from config import config
# name of image file
IMAGE = "imagingVolume.nrrd"

# name of segmentation file
SEGMENTATION = "segMask_tumor.nrrd"

# name of weight indicators
T1 = "t1"
T2 = "t2"
T1C = "t1c"
PD = "pd"

# files you need for image processing
ACCEPTED_FILENAMES = [IMAGE, SEGMENTATION]

# patients to skip, reasons commented
SKIP = ["china-011", #error
        "china-026", #error
        "china-091", #error
        "china-106", #error
        "china-108", #error
        "china-149", #error
        "china-172", #error
        "penn-027", #error
        "02671505", #error
        "55283562", #error
        "02290525", #no data
        "02574281", #no data
        "55162100"] #no data

reader = csv.reader(open(config.CHOP_IDS))
CHOP = {}
for row in reader:
    key = row[1]
    if len(key) == 7:
        key = "0"+key
    if key in CHOP:
        pass
    CHOP[key] = row[0]
