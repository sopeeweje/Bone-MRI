<h1>Deep Learning for evaluation of bone tumors on MRI</h1>
<h2>Training models</h2>
<h3>Setup</h3>
<h4>Config</h4>
Runs can be configured by editing parameters in config.py. This includes:
<ul>
        <li>Epochs/patience</li>
        <li>Train/Validation/Test split</li>
        <li>Image size</li>
        <li>Batch size</li>
        <li>Directory locations</li>
        <li>Cross validation setup (# of folds, train/validation split</li>
</ul>
<h4>Data setup</h4>
<p>Place all raw image data files into the "raw_data" directory. The image and segmentation volumes should be structured as shown in the example data. The example data is real, segmented MRI data from the CHAOS grand challenge (https://chaos.grand-challenge.org/) as the data from our study is not being made publicly available. Each representative subject was randomly given faux age, sex, tumor, location, and malignancy categorization data, described in "features.csv".</p> 
<p>Category "0" refers to tumors that are benign according to the WHO histological categories for bone tumors, "1" is intermediate, and "2" is malignant. In accordance with standard clinical practice, tumors in the intermediate category are assigned as benign for the binary classification task. This is done with the OUTCOME_BIAS parameter in the config file. Changing this parameter to "neg" and rerunning the feature generation will make intermediate tumors group with malignant.</p>
<p>Locations are one-hot encoded according to the categories described in calculate_features.py (all_locations_map, line 48).</p>
<h4>Set up local environment</h4>
Make sure pipenv is installed. Once installed, run:
<pre>
>>> pipenv shell
>>> sh setup.sh
</pre>
<h4>Adjust hyperparameters</h4>
Files - hyperparameters.json, xhyperparameters.json

<h3>Run model</h3>
<h4>Setting up run</h4>
run-model.sh and xrun-model.sh contain the following type of commands:
<pre>pipenv run python run.py --description $1 --model model --form form --label label --hyperparameters hyperparameters.json --split split</pre>
Flags for this command:
<ul>
        <li>--model - Selected model for run ("v1", "v2", etc.)</li>
        <li>--description - unique identifier for run</li>
        <li>--form - input data form ("t1", "t2", etc.)</li>
        <li>--label - label used for points (ex. "outcome_pos")</li>
        <li>--split - split uuid</li>
        <li>--hyperparameters - json with hyperparameters for run</li>
        <li>--trials - number of trials</li>
</ul>
<h4>Executing image data run</h4>
Run <code>sh run.sh $1</code> to execute the below commands:
<pre>
pipenv run python calculate_features.py
pipenv run python preprocess.py
bash run-model.sh $1
</pre>
where $1 is a unique run identifier. Feature calculation and preprocessing should typically only be run during the initial run and upon the introduction of new data.
<h4>Executing clinical data run </h4>
This script takes age, sex, and one-hot encoded location data and trials a number of model types (logistic regression, decision trees, gradient boosting, etc.) to predict malignancy with these clinical features.
<pre>
pipenv run python clinical_data_models.py
</pre>
<h2>Evaluate models</h2>
<h3>Evaluating specific models</h3>
Identify the uuid of the model you would like to evaluate in ouput/results.db. Run the evaluate-specific-model.ipynb notebook.
<h3>Creating ensemble models</h3>
Identify the uuid of the models you would like to would like to include as part of the ensemble in ouput/results.db. Run the evaluate-ensemble.ipynb notebook.
<h3>Comparing ensemble models to expert evaluations</h3>
Utilize evaluate-by-uuids-final.ipynb to create and evaluate ensembles of models trained on individual imaging modalities and the clinical features set and compare them to expert performance. evaluate-ensemble.ipynb contains the classifiers and parameters used to generate each ensemble. Find the desired model uuids in the results.db database.

<h2>Utilize models</h2>
