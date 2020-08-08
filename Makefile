# defaults
.DEFAULT_GOAL := help

# Use bash not sh
SHELL := /bin/bash

REPO_DIR := $(shell git rev-parse --show-toplevel)
VIRTUALENV := $(REPO_DIR)/.venv


.PHONY: venv
venv: ## creates virtualenvironment at the repo's root folder
	python3 -m venv $(VIRTUALENV)
	$(VIRTUALENV)/bin/python -m pip install --upgrade pip wheel setuptools


.PHONY: autoformat
autoformat: ## runs black python formatter on this service's code. Use AFTER make install-*
	# sort imports
	@python3 -m isort --verbose \
		--atomic \
		--recursive \
		--skip-glob */.dvc/* \
		--skip-glob */.venv/* \
		--skip-glob */.vscode/* \
		"$(CURDIR)"
	# auto formatting with black
	@python3 -m black --verbose \
		--exclude "/(\.eggs|\.git|\.hg|\.mypy_cache|\.nox|\.tox|\.venv|\.svn|_build|buck-out|build|dist|migration|client-sdk|generated_code)/" \
		"$(CURDIR)"


.PHONY: convert
convert: ## converts files from python v2 to v3
	2to3 -w $(shell find . -type f -name "*.py")
	# deleting bak files with old version
	rm $(shell find . -type f -name "*.bak")


.PHONY: help
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:
	@echo "usage: make [target] ..."
	@echo ""
	@echo "Targets for '$(notdir $(CURDIR))':"
	@echo ""
	@awk --posix 'BEGIN {FS = ":.*?## "} /^[[:alpha:][:space:]_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""


.PHONY: clean
_GIT_CLEAN_ARGS = -dxf -e .vscode
clean: ## cleans all unversioned files in project and temp files create by this makefile
	# Cleaning unversioned
	@git clean -n $(_GIT_CLEAN_ARGS)
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo -n "$(shell whoami), are you REALLY sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@git clean $(_GIT_CLEAN_ARGS)
