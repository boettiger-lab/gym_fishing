.PHONY: install test tensorboard
# install module such that we can load it from anywhere in the python env

install: 
	pip install -r examples/keras-rl/requirements.txt
	python setup.py sdist bdist_wheel
	pip install -e .

tensorboard: 
	tensorboard --logdir /tmp/tensorboard --port 2223 &

test:	
	python examples/keras-rl/fishing.py


