init:
	pip install -r requirements.txt

test:
	py.test tests

setup:
	python setup.py build_ext --inplace

.PHONY: init test