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
import shutil
from segmentation import calculate_largest_slice, select_slice, bounding_box, crop, resize #file with segmentation functions
from config import config #configuration file
import matplotlib.pyplot as plt
#from path import Path #path = wrapper for os.path
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
    save_path = "/Volumes/external/bone_master/bone_qa/"
    messed_up = []
    
    #available_sequences = {'t1':[],'t2':[],'t1c':[],'pd':[], 'none':[], 't2-t1c':[], 't2-t1':[], 'all_patients':[]}
    seq_by_ID = [["patientID", "t1", "t1c", "t2"]]
    
    print("Loading images...")
    
    if not os.path.isdir(path):
        print("Error: Path does not exist.")
        return
    
    data_all = os.listdir(path) #all sources of data (ex. PENN, China, CHOP)
    total_patients = 0
    num_sources = 0
    for source in data_all:
        num_sources+=1
        if source == '.DS_Store':
            continue
        print("Current source: "+ source)
        data_source = os.listdir(path + "/" + source) #all patients from particular source 
        num_patients = float(len(data_source))
        print("Number of patients: " + str(num_patients))
        completed = 0
        threshold = 10
        
        for data_patient in data_source: #for each patient
            total_patients += 1
            new_input = [data_patient,0,0,0]
            if data_patient == '.DS_Store' or data_patient == '._.DS_Store':#or data_patient in SKIP:
                continue
            print(data_patient)
            patient_views = os.listdir(path + "/" + source + "/" + data_patient)
            #added_views = 0
            t1_dup = 0
            t2_dup = 0
            t1c_dup = 0
            for view in patient_views:
                view_path = path + "/" + source + "/" + data_patient + "/" + view + "/"
                if view == '.DS_Store' or not os.path.isdir(view_path):
                    continue
                
                imageVolume = view_path + "imagingVolume.nrrd"
                segMask = view_path + "segMask_tumor.nrrd"
                segMask_GTV = view_path + "segMask_GTV.nrrd"
                if os.path.isfile(segMask_GTV):
                    os.rename(view_path + "segMask_GTV.nrrd", view_path + "segMask_tumor.nrrd")
                has_volumes = True
                if not (os.path.isfile(imageVolume) and os.path.isfile(segMask)):
                    has_volumes = False
                    nrrds = []
                    for file in os.listdir(view_path):
                        if file.endswith(".nrrd"):
                            nrrds.append(file)
                    print("{}, {}, {}".format(data_patient, view, nrrds))
                    continue
                
                if "t1" in view and has_volumes:
                    if "t1c" in view:
                        new_input[2] = 1
                        try:
                            os.rename(view_path, path + "/" + source + "/" + data_patient + "/t1c")
                        except:
                            os.rename(view_path, path + "/" + source + "/" + data_patient + "/t1c" + "_{}".format(str(t1c_dup)))
                            t1c_dup += 1
                    else:
                        new_input[1] = 1
                        try:
                            os.rename(view_path, path + "/" + source + "/" + data_patient + "/t1")
                        except:
                            os.rename(view_path, path + "/" + source + "/" + data_patient + "/t1" + "_{}".format(str(t1_dup)))
                            t1_dup += 1
                            
                elif "t2" in view and has_volumes:
                    new_input[3] = 1
                    try:
                        os.rename(view_path, path + "/" + source + "/" + data_patient + "/t2")
                    except:
                        os.rename(view_path, path + "/" + source + "/" + data_patient + "/t2" + "_{}".format(str(t2_dup)))
                        t2_dup += 1
                else:
                    pass
                seq_by_ID.append(new_input)
                #try:
                    #if data_patient not in available_sequences[view]:
                        #available_sequences[view].append(data_patient)
                        #added_views += 1 
                #except:
                    #pass
                                        
                #if not (os.path.isfile(imageVolume) and os.path.isfile(segMask)):
                #    continue
                #try:
                #    images = [load_image(imageVolume, segMask, i) for i in range(3)]
                #    plt.imsave(save_path+data_patient+"_"+view+"_"+"0.png", images[0])
                #    plt.imsave(save_path+data_patient+"_"+view+"_"+"1.png", images[1])
                #    plt.imsave(save_path+data_patient+"_"+view+"_"+"2.png", images[2])
                #except:
                #    messed_up.append(data_patient)
                #    continue
                #if not os.path.isfile(save_path+data_patient+"_"+view+"_"+"0.png"): 
                    #plt.imsave(save_path+data_patient+"_"+view+"_"+"0.png", images[0])
                    #plt.imsave(save_path+data_patient+"_"+view+"_"+"1.png", images[1])
                    #plt.imsave(save_path+data_patient+"_"+view+"_"+"2.png", images[2])
            
            #available_sequences['all_patients'].append(data_patient)
            #if not added_views:
                #available_sequences['none'].append(patientID)  
            #completed = completed + 1
            if completed/num_patients*100 > threshold:
                print(str(threshold) + "% complete")
                threshold = threshold + 10
                
    #keys = sorted(available_sequences.keys())
    #t1_available = available_sequences['t1']
    #t1c_available = available_sequences['t1c']
    #t2_available = available_sequences['t2']
    #for patient in t2_available:
       #if patient in t1c_available:
    #        available_sequences['t2-t1c'].append(patient)
    #   if patient in t1_available:
    #        available_sequences['t2-t1'].append(patient)
    #keys = sorted(available_sequences.keys())
    #print("total sources = {}".format(str(num_sources)))
    #print("total patients = {}".format(str(total_patients)))
    #print("%t1 = {}%".format(str(len(available_sequences["t1"])/total_patients*100)))
    #print("%t2 = {}%".format(str(len(available_sequences["t2"])/total_patients*100)))
    #print("%t1c = {}%".format(str(len(available_sequences["t1c"])/total_patients*100)))
    #with open(config.SEQ_AVAIL, "w") as outfile:
    #   writer = csv.writer(outfile, delimiter = ",")
    #   writer.writerow(keys)
    #   writer.writerows(itertools.zip_longest(*[available_sequences[key] for key in keys]))  
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(seq_by_ID)   
    print(messed_up)

#Test code
path = "/Volumes/external/bone_raw"#"/home/user1/Documents/Bone-MRI/bone_raw"
processFiles(path)

