mypy_test:
	mypy --namespace-packages test/examples.py

build_wheels:
	python -m build

publish: build_wheels
	python -m twine upload --repository pypi dist/*
