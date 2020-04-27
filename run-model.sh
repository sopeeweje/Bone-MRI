#pipenv run python run.py --description $1 --model v2 --form t2 --label outcome_pos --hyperparameters hyperparameters_shallow.json --trials 1 --split 40be5ca1-f243-4e59-b35b-69653e7c3f13

#pipenv run python run.py --description $1 --model v2 --form t2 --label outcome_neg --hyperparameters hyperparameters_shallow.json --trials 1 --split 40be5ca1-f243-4e59-b35b-69653e7c3f13

#pipenv run python run.py --description $1 --model v3 --form t2 --label outcome_3 --hyperparameters hyperparameters_shallow.json --trials 1 --split 40be5ca1-f243-4e59-b35b-69653e7c3f13

pipenv run python run.py --description $1 --model v2 --form t1 --label outcome_pos --hyperparameters hyperparameters_long.json --trials 1 --split 40be5ca1-f243-4e59-b35b-69653e7c3f13

pipenv run python run.py --description $1 --model v2 --form t1 --label outcome_neg --hyperparameters hyperparameters_long.json --trials 1 --split 40be5ca1-f243-4e59-b35b-69653e7c3f13

pipenv run python run.py --description $1 --model v3 --form t1 --label outcome_3 --hyperparameters hyperparameters_long.json --trials 1 --split 40be5ca1-f243-4e59-b35b-69653e7c3f13

