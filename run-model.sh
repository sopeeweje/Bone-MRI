pipenv run python run.py --description $1 --model v2 --form t2 --label outcome_pos --hyperparameters hyperparameters_long.json --trials 3 --split 484f5484-9d32-42f9-b08d-42448407eb45

pipenv run python run.py --description $1 --model v2 --form t2 --label outcome_neg --hyperparameters hyperparameters_long.json --trials 3 --split e4357770-dc4b-4472-8dc9-310032bdade7

pipenv run python run.py --description $1 --model v3 --form t2 --label outcome_3 --hyperparameters hyperparameters_long.json --trials 3 --split 426505f6-92b2-4724-b624-b11088d76527

pipenv run python run.py --description $1 --model v2 --form t1c --label outcome_pos --hyperparameters hyperparameters_long.json --trials 3 --split 484f5484-9d32-42f9-b08d-42448407eb45

pipenv run python run.py --description $1 --model v2 --form t1c --label outcome_neg --hyperparameters hyperparameters_long.json --trials 3 --split e4357770-dc4b-4472-8dc9-310032bdade7

pipenv run python run.py --description $1 --model v3 --form t1c --label outcome_3 --hyperparameters hyperparameters_long.json --trials 3 --split 426505f6-92b2-4724-b624-b11088d76527

