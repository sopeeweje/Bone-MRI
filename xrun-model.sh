
pipenv run python xrun.py --trials 10 --description $1 --model v2 --form t1 --label outcome --hyperparameters xhyperparameters.json --split 62e0fad6-783f-4453-af2f-a52b241f1188
pipenv run python xrun.py --trials 10 --description $1 --model v2 --form t2 --label outcome --hyperparameters xhyperparameters.json --split 62e0fad6-783f-4453-af2f-a52b241f1188
pipenv run python xrun.py --trials 10 --description $1 --model v2 --form features --label outcome --hyperparameters xhyperparameters.json --split 62e0fad6-783f-4453-af2f-a52b241f1188