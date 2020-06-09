<h1>Deep Learning for evaluation of bone tumors on MRI</h1>

<h2>About</h2>

<h2>Training models</h2>
<h3>Setup</h3>
<h4>Quality assurance</h4>
Place all raw files into a single directory. Run <b>qa.py</b>. This file will generate a directory of png files to manually scan through to look for abnormalities. Place the IDs of subjects that have abnormal images into the "SKIP" list in <b>filenames.py</b>.
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
<b>run-model.sh</b> and <b>xrun-model.sh</b> contain the following type of commands:
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
<h4>Executing straight run</h4>
Run <code>sh run.sh $1</code> to execute the below commands:
<pre>
pipenv run python calculate_features.py
pipenv run python preprocess.py
bash run-model.sh $1
</pre>
where $1 is a unique run identifier.
<h4>Executing cross-validation run </h4>
Run <code>sh xrun.sh $1</code> to execute the below commands:
<pre>
pipenv run python calculate_features.py
pipenv run python preprocess.py
bash xrun-model.sh $1
</pre>
where $1 is a unique run identifier.

<h3>Evaluate models</h3>
**evaluate.py**
***run.py*** and ***xrun.py*** import evaluate from ***evaluate.py***. This is a function for evaluating the model with the test set. The get_results function is run within ***run.py*** and ***xrun.py***; the other functions can be executed by manually extracting data from the db as you perform analyses (or maybe there's a notebook for it?).
ssh jupyter notebook
https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/
https://amber-md.github.io/pytraj/latest/tutorials/remote_jupyter_notebook
**Results**
app
api
config
db
error_analysis
stacked_data

<h2>Utilize models</h2>
