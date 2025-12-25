.PHONY: install build publish clean test

install:
	poetry install

build:
	poetry build

publish:
	poetry publish

test:
	poetry run pytest

clean:
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +


