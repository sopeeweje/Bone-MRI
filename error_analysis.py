import numpy
import pandas
import nrrd
from config import config
from filenames import T1, T2, IMAGE, SEGMENTATION, ENHANCED_IMAGE, ENHANCED_SEGMENTATION
pre_files = config.PREPROCESS_FEATURES
post_files = config.PREPROCESS_FEATURES_POST

SHAPES = """SHAPES
t1: {}
t1 seg: {}
t2: {}
t2 seg: {}"""

FILES = """using files:
t1: {}
t1 seg: {}
t2: {}
t2 seg: {}"""

EQUALITY = """equal files:
t1==t2: {}
t1_seg==t2_seg: {}
t1==t1_seg: {}
t1==t2_seg: {}
t2==t1_seg: {}
t2==t2_seg: {}"""

ERROR = """
Error reading: {}
t1: {}
t1_seg: {}
t2: {}
t2_seg: {}
error:
{}"""

ENHANCED_SHAPES = """SHAPES
enhanced: {}
enhanced seg: {}"""

ENHANCED_FILES = """using files:
enhanced: {}
enhanced seg: {}"""

ENHANCED_EQUALITY = """equal files:
enhanced==enhanced_seg: {}"""

ENHANCED_ERROR = """
Error reading: {}
enhanced: {}
enhanced_seg: {}
error:
{}"""

def check(index, t1, t1_seg, t2, t2_seg, enhanced, enhanced_seg):
    try:
        t1_nrrd, _ = nrrd.read(t1)
        t1_seg_nrrd, _ = nrrd.read(t1_seg)
        t2_nrrd, _ = nrrd.read(t2)
        t2_seg_nrrd, _ = nrrd.read(t2_seg)
        if (t1_nrrd.shape != t1_seg_nrrd.shape) or (t2_nrrd.shape != t2_seg_nrrd.shape):
            print() #extra space
            print("accession:{}".format(index))
            print(FILES.format(t1, t1_seg, t2, t2_seg))
            print(SHAPES.format(t1_nrrd.shape, t1_seg_nrrd.shape, t2_nrrd.shape, t2_seg_nrrd.shape))
            print(EQUALITY.format(
                numpy.array_equal(t1_nrrd, t2_nrrd),
                numpy.array_equal(t1_seg_nrrd, t2_seg_nrrd),
                numpy.array_equal(t1_nrrd, t1_seg_nrrd),
                numpy.array_equal(t1_nrrd, t2_seg_nrrd),
                numpy.array_equal(t2_nrrd, t1_seg_nrrd),
                numpy.array_equal(t2_nrrd, t2_seg_nrrd),
            ))
    except Exception as e:
        print(ERROR.format(index, t1, t1_seg, t2, t2_seg, e))

    try:
        if type(enhanced) is not str or type(enhanced_seg) is not str:
            return
        enhanced_nrrd, _ = nrrd.read(enhanced)
        enhanced_seg_nrrd, _ = nrrd.read(enhanced_seg)
        if enhanced_nrrd.shape != enhanced_seg_nrrd.shape:
            print()
            print("enhanced accession:{}".format(index))
            print(ENHANCED_FILES.format(enhanced, enhanced_seg))
            print(ENHANCED_SHAPES.format(enhanced_nrrd.shape, enhanced_seg_nrrd.shape))
            print(ENHANCED_EQUALITY.format(
                numpy.array_equal(enhanced_nrrd, enhanced_seg_nrrd),
            ))
    except Exception as e:
        print(ENHANCED_ERROR.format(index, enhanced, enhanced_seg, e))

if __name__ == '__main__':
    f = pandas.read_pickle(pre_files)
    for index, row in f.iterrows():
            t1 = row.to_frame().loc["path", T1, IMAGE][0]
            t1_seg = row.to_frame().loc["path", T1, SEGMENTATION][0]
            t2 = row.to_frame().loc["path", T2, IMAGE][0]
            t2_seg = row.to_frame().loc["path", T2, SEGMENTATION][0]
            enhanced = row.to_frame().loc["path", T1, ENHANCED_IMAGE][0]
            enhanced_seg = row.to_frame().loc["path", T1, ENHANCED_SEGMENTATION][0]
            check(index, t1, t1_seg, t2, t2_seg, enhanced, enhanced_seg)
    f = pandas.read_pickle(post_files)
    for index, row in f.iterrows():
            t1 = row.to_frame().loc["path", T1, IMAGE][0]
            t1_seg = row.to_frame().loc["path", T1, SEGMENTATION][0]
            t2 = row.to_frame().loc["path", T2, IMAGE][0]
            t2_seg = row.to_frame().loc["path", T2, SEGMENTATION][0]
            enhanced = row.to_frame().loc["path", T1, ENHANCED_IMAGE][0]
            enhanced_seg = row.to_frame().loc["path", T1, ENHANCED_SEGMENTATION][0]
            check(index, t1, t1_seg, t2, t2_seg, enhanced, enhanced_seg)
