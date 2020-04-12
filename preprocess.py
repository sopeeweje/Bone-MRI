import os
import argparse
import nrrd
import pandas
import numpy as np
import SimpleITK as sitk

from config import config

from filenames import IMAGE, SEGMENTATION, T1, T2, T1C

def n4_bias_correction(image):
    import ants
    as_ants = ants.from_numpy(image)
    corrected = ants.n4_bias_field_correction(as_ants)
    return corrected.numpy()

def registration(reference, image, segmentation):
    import ants
    reference_as_ants = ants.from_numpy(reference)
    image_as_ants = ants.from_numpy(image)
    output = ants.registration(reference_as_ants, image_as_ants)
    registered_image = output.get("warpedmovout")
    segmentation_as_ants = ants.from_numpy(segmentation)
    registered_segmentation = ants.apply_transforms(reference_as_ants, segmentation_as_ants, output.get("fwdtransforms"))
    registered_segmentation = registered_segmentation.numpy()
    registered_segmentation[registered_segmentation > 0] = 1
    return registered_image.numpy(), registered_segmentation

def intensity_normalization(reference, image):
    reference_as_sitk = sitk.GetImageFromArray(reference)
    image_as_sitk = sitk.GetImageFromArray(image)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(128)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    output = matcher.Execute(image_as_sitk, reference_as_sitk)
    return np.array(sitk.GetArrayViewFromImage(output))

def preprocess_pack(reference, images_w_segmentations, use_n4_bias=False, use_registration=False):
    final_images = list()
#    if use_n4_bias:
#        reference = n4_bias_correction(reference)
    for (image, segmentation) in images_w_segmentations:
        image = image.astype(float)
#        if use_n4_bias:
#            image = n4_bias_correction(image)
#        if use_registration:
#            image, segmentation = registration(reference, image, segmentation)
        image = intensity_normalization(reference, image)
        final_images.append((image, segmentation))
    return reference, final_images

def run(files, out, use_n4_bias=False, use_registration=False):
    f = pandas.read_pickle(files)
    
    for index, row in f.iterrows():
        print("working on {} {}".format(index, "-" * 40))
        
        #for using T1
        print("For using T1 input")
        try:
            if not ("{}-{}-{}".format(index, T1, IMAGE) in os.listdir(config.PREPROCESSED_DIR)):
                t1 = row.to_frame().loc["path", T1, IMAGE][0]
                t1_seg = row.to_frame().loc["path", T1, SEGMENTATION][0]
                print("""using files:
                t1: {}
                t1 seg: {}
                """.format(t1, t1_seg))
                t1_nrrd, _ = nrrd.read(t1)
                t1_seg_nrrd, _ = nrrd.read(t1_seg)
                print("""SHAPES
                      t1: {}
                      t1 seg: {}""".format(t1_nrrd.shape, t1_seg_nrrd.shape))
                nrrd.write(os.path.join(out, "{}-{}-{}".format(index, T1, IMAGE)), t1_nrrd)
                nrrd.write(os.path.join(out, "{}-{}-{}".format(index, T1, SEGMENTATION)), t1_seg_nrrd)
                print("#" * 40)
            else:
                print("This image has been preprocessed!")
        except Exception as e:
            print()
            print("Exception occurred for: {}\n{}".format(index, e))
            print("No T1 available")
            print("#" * 40)
        
        #for using T1C       
        print("For using T1C input")
        try:
            if not ("{}-{}-{}".format(index, T1C, IMAGE) in os.listdir(config.PREPROCESSED_DIR)):
                t1c = row.to_frame().loc["path", T1C, IMAGE][0]
                t1c_seg = row.to_frame().loc["path", T1C, SEGMENTATION][0]
                print("""using files:
                t1c: {}
                t1c seg: {}
                """.format(t1c, t1c_seg))
                t1c_nrrd, _ = nrrd.read(t1c)
                t1c_seg_nrrd, _ = nrrd.read(t1c_seg)
                print("""SHAPES
                      t1c: {}
                      t1c seg: {}""".format(t1c_nrrd.shape, t1c_seg_nrrd.shape))
                nrrd.write(os.path.join(out, "{}-{}-{}".format(index, T1C, IMAGE)), t1c_nrrd)
                nrrd.write(os.path.join(out, "{}-{}-{}".format(index, T1C, SEGMENTATION)), t1c_seg_nrrd)
                print("#" * 40)
            else:
                print("This image has been preprocessed!")
        except Exception as e:
            print()
            print("Exception occurred for: {}\n{}".format(index, e))
            print("No T1C available")
            print("#" * 40)
        
        #for using T2
        print("For using T2 input")
            #try:
        if not ("{}-{}-{}".format(index, T2, IMAGE) in os.listdir(config.PREPROCESSED_DIR)):
            t1 = row.to_frame().loc["path", T1, IMAGE][0]
            t1_seg = row.to_frame().loc["path", T1, SEGMENTATION][0]
            t2 = row.to_frame().loc["path", T2, IMAGE][0]
            t2_seg = row.to_frame().loc["path", T2, SEGMENTATION][0]
            
            print("""using files:
                    t1: {}
                    t1 seg: {}
                    t2: {}
                    t2 seg: {}
                    """.format(t1, t1_seg, t2, t2_seg))

            t1_nrrd, _ = nrrd.read(t1)
            t1_seg_nrrd, _ = nrrd.read(t1_seg)
            t2_nrrd, _ = nrrd.read(t2)
            t2_seg_nrrd, _ = nrrd.read(t2_seg)
            print("""START SHAPES
                    t2: {}
                    t2 seg: {}
                """.format(t2_nrrd.shape, t2_seg_nrrd.shape))

            _, out_t2 = preprocess_pack(t1_nrrd, [(t2_nrrd, t2_seg_nrrd)], use_n4_bias, use_registration)
            out_t2_image = out_t2[0][0]
            out_t2_seg = out_t2[0][1]

            print("""END SHAPES
                    t2: {}
                    t2 seg: {}
                """.format(out_t2_image.shape, out_t2_seg.shape))
            nrrd.write(os.path.join(out, "{}-{}-{}".format(index, T2, IMAGE)), out_t2_image)
            nrrd.write(os.path.join(out, "{}-{}-{}".format(index, T2, SEGMENTATION)), out_t2_seg)
            print("#" * 80)
        else:
            print("This image has been preprocessed!")
        #except Exception as e:
        #print()
        #print("Exception occurred for: {}\n{}".format(index, e))
        #print("No T2 available, or no T1 available for registration")
        #print("#" * 80)
            
#continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n4', action='store_true', help="use n4 bias correction")
    parser.add_argument('--registration', action='store_true', help="use registration")
    parser.add_argument(
        '--preprocess',
        type=str,
        default=config.PREPROCESS,
        help='preprocess file')
    parser.add_argument(
        '--out',
        type=str,
        default=config.PREPROCESSED_DIR,
        help='output')
    FLAGS, unparsed = parser.parse_known_args()
    run(FLAGS.preprocess, FLAGS.out, FLAGS.n4, FLAGS.registration)
