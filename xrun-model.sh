#pipenv run python xrun.py --description $1 --model v1 --form t2 --label outcome_pos --hyperparameters xhyperparameters.json --split 84a64c17-fe3e-440c-aaaf-e1bd5b02576f

#pipenv run python xrun.py --description $1 --model v1 --form t1 --label outcome_pos --hyperparameters xhyperparameters.json --split 84a64c17-fe3e-440c-aaaf-e1bd5b02576f

pipenv run python xrun.py --description $1 --model v2 --form t2 --label outcome_pos --hyperparameters xhyperparameters.json --split 84a64c17-fe3e-440c-aaaf-e1bd5b02576f

#pipenv run python xrun.py --description $1 --model v2 --form t1 --label outcome_pos --hyperparameters xhyperparameters.json --split 84a64c17-fe3e-440c-aaaf-e1bd5b02576f

#pipenv run python xrun.py --description $1 --model v2 --form t2-features --label outcome --hyperparameters xhyperparameters.json --split a5bf8f5a-5fb2-4105-804b-7b14020f0b87

#pipenv run python xrun.py --trials 10 --description $1 --model v2 --form t2-features --label outcome --hyperparameters xhyperparameters.json --split
