<h1>Deep Learning for evaluation of bone tumors on MRI</h1>

<h2>About</h2>

<h2>Training models</h2>
        <h3>Quality Assurance</h3>
        1. Place all raw files (folders for each patient containing image volumes of various sequences) into one directory
        2. Run ***qa.py***. This file will generate a directory of png files to manually scan through to look for       abnormalities. Place the IDs of subjects that have abnormal images into the "SKIP" list in ***filenames.py***.
        <h3>Set up local environment</h3>
        **setup.sh**
        1. Make sure pipenv is installed.
        2. Run pipenv shell
        3. Run setup.sh
        <h3>Adjusting hyperparameters</h3>
        <h3>Run model</h3>
**run.sh/xrun.sh**
1. Edit ***config.py*** file to reconfigure where the appropriate directories are.
2. Run ***calculate_features.py***. This file will create csv's for training and preprocessing in a sub-directory called "features". The inputs are derived from bone_features.csv in the bone_master directory. In bone_features.csv, there is currently only two columns - patientIDs and category (benign, malignant, intermediate). At this point the model will be trained only with image features; if at a future point clinical features such as age and sex become available, add them into bone_features.csv as well as "bone tumors", a spreadsheet simply for subject tracking that is not used for model generation. Given that age and sex are not being used, there are a number of lines commented out of ***calculate_features.py*** that can be activated if those features are eventually used. In addition, another column called "sort" can be added to bone_features.csv with "train", "test", and "validation" as possible values; this can be used if it is necessary to sort particular subjects into certain datasets. Otherwise it will be randomized.
3. ***calculate_features.py*** imports a Config object from ***config.py***. The Config object contains properties that are paths to the various relevant directories and files. It also imports from ***filenames.py*** a list of files to skip because abnormalities were found during QA and "ACCEPTED_FILENAMES" (imagingVolume.nrrd or segMask_tumor.nrrd [segmented]). calculate_volume is imported from ***segmentation.py*** to add a tumor volume feature.
4. In the "features" function, images classified as intermediate can be biased towards benign or malignant (doing a binary analysis for now). *Change the int_bias parameter to "pos" to bias intermediate to benign or "neg" to bias intermediate to malignant*.
5. Run **preprocess.py**. This file normalizes image intensity, performs N4 bias correction, (https://simpleitk.readthedocs.io/en/latest/Examples/N4BiasFieldCorrection/Documentation.html), and performs image registration. It is currently configured not to perform N4 bias correction and image registration (need to debug this). In ***run.sh*** add --n4 and --registration flags to preprocess perform these functions. It also checks that each subject has T1- and T2-weighted images that have been segmented.

**run-model.sh**
***run-model.sh*** is executed in ***run.sh***, featuring the ***run.py*** file. Key functionalities of ***run.py***
1. Generates a random run id/split id using the uuid4() function unless the split id is specified (which you should specify to make sure you're running on the same datasets as you refine your model). 
2. Imports models from ***models.py***. The currently available models are v1 (***v1.py***, ResNet) and v2 (***v2.py***, EfficientNet).
3. Imports data_gen from ***data.py***. Note that clinical features age and sex are currently suppressed. This file formats a number of different data inputs - using T1 images only, using T2 images only, using T1 + clinical features, using T2 + clinical features, using T1 and T2 images, using just clinical features, using everything. It also sorts the data into training, testing, and validation sets; it can take presorts (from bone_features.csv) or randomly sorted according to designated fractions.
4. Explode_parameters function. Creates a dictionary containing each combination of hyperparameters to be tested. A single execution of ***run.py*** runs each possible combination.

**Running in terminal**
nohup sh run.sh DATE_TIME_ARCHITECTURE_DATA > my.log 2>&1 & #Executing run in background
echo $! > save_pid.txt #Saving process ID for run so you can kill it necessary (kill -9 processID). Alternatively run killall -u sopeeweje to make sure you get everything.
ssh jupyter notebook
https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/
https://amber-md.github.io/pytraj/latest/tutorials/remote_jupyter_notebook

**xrun.sh**

**evaluate.py**
***run.py*** and ***xrun.py*** import evaluate from ***evaluate.py***. This is a function for evaluating the model with the test set. The get_results function is run within ***run.py*** and ***xrun.py***; the other functions can be executed by manually extracting data from the db as you perform analyses (or maybe there's a notebook for it?).
        
**v1.py and v2.py**

**Running the Program**
The main driver of this program is in run.sh and xrun.sh 
Pre-processing is done through calculate_features.py and preprocess.py
Running the models is done through run-model.sh

To start the cross-validation, run xrun.sh NameOfRun
Each NameOfRun should be unique as possible

**Results**
app
api
config
db
error_analysis
stacked_data
