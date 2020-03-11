pipenv run python run.py --description $1 --model v2 --form t1c-features --label outcome --hyperparameters hyperparameters.json #--split 27863f52-d15d-4cc1-be66-155872f20d6b 

#change form to get different combos of input features, can be imaging alone, clinical alone, or both

#pipenv run python run.py --description $1 --model v2 --form t2 --label outcome --hyperparameters hyperparameters.json --split 6f91b43f-d6cd-42dc-9eb6-8178f4ea8cc1
#pipenv run python run.py --description $1 --model v2 --form features --label outcome --hyperparameters hyperparameters.json --split 6f91b43f-d6cd-42dc-9eb6-8178f4ea8cc1
