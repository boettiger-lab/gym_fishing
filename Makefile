.PHONY: install test env
# install module such that we can load it from anywhere in the python env

install: 
	python setup.py sdist bdist_wheel
	pip install -e .


test:	
	python examples/keras-rl/fishing.py


