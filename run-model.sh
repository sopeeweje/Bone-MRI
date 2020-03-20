pipenv run python run.py --description $1 --model v2 --form t1c-features --label outcome --hyperparameters hyperparameters.json --split 4860bb0e-11d4-493c-b9b0-bb8822952f71 #27863f52-d15d-4cc1-be66-155872f20d6b #b888e896-e151-4224-8b4e-12d36f6307c8
pipenv run python run.py --description $1 --model v3 --form t1c-features --label outcome --hyperparameters hyperparameters.json --split 4860bb0e-11d4-493c-b9b0-bb8822952f71

#change form to get different combos of input features, can be imaging alone, clinical alone, or both

#pipenv run python run.py --description $1 --model v2 --form t2 --label outcome --hyperparameters hyperparameters.json --split 6f91b43f-d6cd-42dc-9eb6-8178f4ea8cc1
#pipenv run python run.py --description $1 --model v2 --form features --label outcome --hyperparameters hyperparameters.json --split 6f91b43f-d6cd-42dc-9eb6-8178f4ea8cc1
