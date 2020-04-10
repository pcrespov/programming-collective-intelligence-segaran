.venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip wheel setuptools

.PHONY: devenv
devenv: .venv
	.venv/bin/pip install -r requirements.txt
