#Get each patient file. Get different views from each file. Get 3 axes from each view

#t1 - T1
#t1c - T1 + contrast
#t2 - T2
#pwi - perfusion-weight imaging

#ax - axial
#cor - coronal
#sag - sagittal

#sat - saturated
#fs - fat saturated
#un - neither

#make a nested dictionary - key = patientID, value = dictionary with keys as each possible combo of above descriptors, value is associated image

import nrrd #libary to read and write NRRD files into and from numpy arrays (nrrd - multi-dimensional image file type)
import os #interacting with terminal
from segmentation import calculate_largest_slice, select_slice, bounding_box, crop, resize #file with segmentation functions
from config import config #configuration file
import matplotlib.pyplot as plt
from path import Path #path = wrapper for os.path
from filenames import SKIP
import csv
import itertools

def deleteUselessFiles(path):
    """
        Param: file path
        Function: Remove all files that are not .nrrd files
        Returns: None
    """
    d = Path(path)
    allList = d.walkfiles('*')
    for file in allList:
        if not file.endswith(".nrrd"):
            file.remove()    
    return


def load_image(image_path, segmentation_path, axis, verbose=False):
    """
        Param: path to image, path to image segmentation
        Function: Load, resize, and mask image by the segmentation of its largest slice
        Returns: List masked images representing each of the 3 views
    """
    image, _ = nrrd.read(image_path)
    segmentation, _ = nrrd.read(segmentation_path)
    if verbose:
        print("""
        image: {}
        seg: {}
        """.format(image.shape, segmentation.shape))

    largest_plane = calculate_largest_slice(segmentation, axis) #find largest cross section
    image, segmentation = select_slice(image, segmentation, largest_plane, axis) #image and segmentation pattern of the largest cross section
    
    bounds = bounding_box(segmentation) #Bound image by largest slice
    image, segmentation = crop(image, segmentation, bounds) #Crop image to fit bounds
    
    masked = image * segmentation #Mask everything not of interest (#keep things in segment 1, lose things in segment 0)
    masked = resize(masked, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #Standardize masked image to given size
    
    return masked
    
def processFiles(path):
    """
    Param: path to raw images
    Returns: all png's to folder to look through for QA
    """
    save_path = "/Volumes/external/bone_master/bone_qa"
    chopIDs = open("/Volumes/external/bone_master/chopIDs.csv")
    chopIDs_list = list(csv.reader(chopIDs))
    chopIDs_dict = {}
    
    for combo in chopIDs_list:
        chopIDs_dict[combo[1]] = combo[0]
    messed_up = []
    
    available_sequences = {'t1':[],'t2':[],'t1c':[],'pd':[]}
    
    print("Loading images...")
    
    if not os.path.isdir(path):
        print("Error: Path does not exist.")
        return
    
    data_all = os.listdir(path) #all sources of data (ex. PENN, China, CHOP)
    
    for source in data_all:
        if source == '.DS_Store':
            continue
        print("Current source: "+ source)
        data_source = os.listdir(path + "/" + source) #all patients from particular source 
        num_patients = float(len(data_source))
        print("Number of patients: " + str(num_patients))
        completed = 0
        threshold = 10
        
        for data_patient in data_source: #for each patient
            if data_patient == '.DS_Store' or data_patient in SKIP:
                continue
            
            if source == "CHOP":
                patientID = chopIDs_dict[data_patient]
            elif source == "Penn":
                patientID = data_patient
            else:
                patientID = data_patient.replace("bone","").replace("-","")
                
            print("Patient: " + patientID)
            patient_views = os.listdir(path + "/" + source + "/" + data_patient)
            
            for view in patient_views:
                if view == '.DS_Store':
                    continue
                
                ''.join(e for e in view if e.isalnum()) #use for non-CHOP files
                view = view.split(" ")[0]
                
                try:
                    if patientID not in available_sequences[view]:
                        available_sequences[view].append(patientID) 
                except:
                    print(view)
                
#                new_image_path = save_path#+"/"+patientID+"/"+view
                #os.makedirs(new_image_path)
                
#                imageVolume = path + "/" + source + "/" + data_patient + "/" + view + "/" + "imagingVolume.nrrd"
#                segMask = path + "/" + source + "/" + data_patient + "/" + view + "/" + "segMask_tumor.nrrd"
#                if not (os.path.isfile(imageVolume) and os.path.isfile(segMask)):
#                    continue
#                try:
#                    images = [load_image(imageVolume, segMask, i) for i in range(3)]
#                except:
#                    messed_up.append(data_patient)
#                    continue
#              
#                if source != "CHOP":
#                # use for non CHOP files
#                    plt.imsave(new_image_path+"/"+patientID+"_"+view+"_"+"0.png", images[0])
#                    plt.imsave(new_image_path+"/"+patientID+"_"+view+"_"+"1.png", images[1])
#                    plt.imsave(new_image_path+"/"+patientID+"_"+view+"_"+"2.png", images[2])
#                
#                if source == "CHOP":
#                # use for CHOP files
#                    plt.imsave(new_image_path+"/"+patientID+"_"+view.split(" ")[0]+"_"+"0.png", images[0])
#                    plt.imsave(new_image_path+"/"+patientID+"_"+view.split(" ")[0]+"_"+"1.png", images[1])
#                    plt.imsave(new_image_path+"/"+patientID+"_"+view.split(" ")[0]+"_"+"2.png", images[2])
#                completed = completed + 1
#                if completed/num_patients*100 > threshold:
#                    print(str(threshold) + "% complete")
#                    threshold = threshold + 10
#                
    keys = sorted(available_sequences.keys())
    with open(config.SEQ_AVAIL, "w") as outfile:
       writer = csv.writer(outfile, delimiter = ",")
       writer.writerow(keys)
       writer.writerows(itertools.zip_longest(*[available_sequences[key] for key in keys]))
            

#Test code
path = "/Volumes/external/bone_master/bone_raw"
processFiles(path)

