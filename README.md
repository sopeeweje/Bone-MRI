<h1>Deep Learning for evaluation of bone tumors on MRI</h1>

<h2>About</h2>

<h2>Training models</h2>
<h3>Setup</h3>
<h4>Quality assurance</h4>
Place all raw files into a single directory. Run qa.py. This file will generate a directory of png files to manually scan through to look for abnormalities. Place the IDs of subjects that have abnormal images into the "SKIP" list in filenames.py.
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
<h3>Evaluate models</h3>
<h4>Evaluating specific models</h4>

<h4>Creating ensemble models</h4>

<h4>Comparing ensemble models to expert evaluations</h4>
Utilize evaluate-by-uuids-final.ipynb to create and evaluate ensembles of models trained on individual imaging modalities and the clinical features set and compare them to expert performance. evaluate-ensemble.ipynb contains the classifiers and parameters used to generate each ensemble. Find the desired model uuids in the results.db database.

<h2>Utilize models</h2>
