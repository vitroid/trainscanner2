.PHONY: install build publish clean test push

VERSION := $(shell grep '^version =' pyproject.toml | cut -d '"' -f 2)

install:
	poetry install

build:
	poetry build

publish: build
	poetry publish

push:
	@if [ -z "$$(git tag -l v$(VERSION))" ]; then \
		echo "Creating tag v$(VERSION)..."; \
		git tag -a v$(VERSION) -m "Release v$(VERSION)"; \
	else \
		echo "Tag v$(VERSION) already exists."; \
	fi
	git push origin $$(git rev-parse --abbrev-ref HEAD) --tags

test:
	poetry run pytest

clean:
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +


