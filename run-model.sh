pipenv run python run.py --description $1 --model v2 --form t2 --label outcome_pos --hyperparameters hyperparameters_long.json --trials 3 --split fa6c10e6-a6d7-478e-9a3e-e3925f448f15

pipenv run python run.py --description $1 --model v2 --form t2 --label outcome_neg --hyperparameters hyperparameters_long.json --trials 3 --split 0bab6177-e60c-4a85-8deb-e1cb5526af4f

pipenv run python run.py --description $1 --model v3 --form t2 --label outcome_3 --hyperparameters hyperparameters_long.json --trials 3 --split 4dc253c8-34ce-4273-b495-27d2885ebadf

pipenv run python run.py --description $1 --model v2 --form t1c --label outcome_pos --hyperparameters hyperparameters_long.json --trials 3 --split fa6c10e6-a6d7-478e-9a3e-e3925f448f15

pipenv run python run.py --description $1 --model v2 --form t1c --label outcome_neg --hyperparameters hyperparameters_long.json --trials 3 --split 0bab6177-e60c-4a85-8deb-e1cb5526af4f

pipenv run python run.py --description $1 --model v3 --form t1c --label outcome_3 --hyperparameters hyperparameters_long.json --trials 3 --split 4dc253c8-34ce-4273-b495-27d2885ebadf

