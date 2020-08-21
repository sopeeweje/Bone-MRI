#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:00:47 2020

@author: Sope
"""
import os
import shutil

path = "/Volumes/external/test set/"
patients = range(2,94)
patient_num = 1
for patient in patients:
    # try:
    #     if patient == '.DS_Store' or patient == '._.DS_Store':
    #         continue
    #     t1_path = "{}{}/t1/".format(path,patient)
    #     t1_volume = t1_path + "imagingVolume.nrrd"
    #     os.rename(t1_volume, t1_path+"t1_patient{}.nrrd".format(str(patient_num)))
    #     shutil.move(t1_path+"t1_patient{}.nrrd".format(str(patient_num)), "{}{}/".format(path,patient))
    #     shutil.rmtree(t1_path)
    # except:
    #     pass#os.rename("{}{}/t1_patient{}".format(path,patient, str(patient_num)), "{}{}/t1_patient{}.nrrd".format(path,patient, str(patient_num)))
    
    # try:
    #     t2_path = "{}{}/t2/".format(path,patient)
    #     t2_volume = t2_path + "imagingVolume.nrrd"
    #     os.rename(t2_volume, t2_path+"t2_patient{}.nrrd".format(str(patient_num)))
    #     shutil.move(t2_path+"t2_patient{}.nrrd".format(str(patient_num)), "{}{}/".format(path,patient))
    #     shutil.rmtree(t2_path)
    # except:
    #     pass#os.rename("{}{}/t2_patient{}".format(path,patient,str(patient_num)), "{}{}/t2_patient{}.nrrd".format(path,patient, str(patient_num)))
        
    #if os.path.isdir("{}{}/t1c/".format(path,patient)):
    #    shutil.rmtree("{}{}/t1c/".format(path,patient))
        
    #if os.path.isdir("{}{}/pd/".format(path,patient)):
    #    shutil.rmtree("{}{}/pd/".format(path,patient)) 
    print(patient)
    shutil.move("{}{}{}".format(path,"patient",patient), "{}{}{}".format(path, "patient_", str(patient-1)))
    