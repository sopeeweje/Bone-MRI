#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:01:28 2020

@author: Sope
"""
import nrrd
from preprocess import preprocess_pack

t1_ref_img, _ = nrrd.read("/Volumes/external/bone_raw/CHOP/bone-penn-332/t1/imagingVolume.nrrd")
t1_ref_seg, _ = nrrd.read("/Volumes/external/bone_raw/CHOP/bone-penn-332/t1/segMask_tumor.nrrd")

t1c_ref_img, _ = nrrd.read("/Volumes/external/bone_raw/CHOP/bone-penn-332/t1c/imagingVolume.nrrd")
t1c_ref_seg, _ = nrrd.read("/Volumes/external/bone_raw/CHOP/bone-penn-332/t1c/segMask_tumor.nrrd")

t2_ref_img, _ = nrrd.read("/Volumes/external/bone_raw/CHOP/bone-penn-332/t2/imagingVolume.nrrd")
t2_ref_seg, _ = nrrd.read("/Volumes/external/bone_raw/CHOP/bone-penn-332/t2/segMask_tumor.nrrd")

t2_nrrd, _ = nrrd.read("/Volumes/external/bone_raw/Penn/bone-penn-064/t2/imagingVolume.nrrd")
t2_seg_nrrd, _ = nrrd.read("/Volumes/external/bone_raw/Penn/bone-penn-064/t2/segMask_tumor.seg.nrrd")
image, seg = preprocess_pack(t2_ref_img, t2_ref_seg, t2_nrrd, t2_seg_nrrd, use_n4_bias=True, use_registration=False)

nrrd.write("preprocess_test_image.nrrd", image)
nrrd.write("preprocess_test_seg.nrrd", seg)
