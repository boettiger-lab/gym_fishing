.PHONY: install test env
# install module such that we can load it from anywhere in the python env

install: 
	python setup.py sdist bdist_wheel
	pip install -e .

env: 
	source ~/.virtualenvs/tf1.14/bin/activate

test:	
	python examples/keras-rl/fishing.py


