.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:        ## Help in the makefile
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

clean: 		## Remove build artifacts and set up environment
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	rm -f uv.lock
	rm -rf .venv/
	rm -rf lib/
	rm -rf lib64/
	rm -rf bin/
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr .ruff_cache/
	rm -fr .mypy_cache/

	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.11
	uv venv
	uv lock
	uv add pandas openpyxl numpy matplotlib statsmodels bokeh scikit-image scikit-learn patsy plotly numba seaborn pydantic tqdm
	uv add --dev flake8 tox coverage Sphinx twine pytest pytest-runner pytest-cov pytest-xdist pre-commit black isort
	uv add --dev pandas-stubs matplotlib-stubs plotly-stubs tqdm-stubs mypy

	uvx pre-commit install
	uvx pre-commit
	uvx pre-commit autoupdate

	uvx mypy --install-types

check:  # if the first command gives a return, then stage those files, then run pre-commit
	git update-index --refresh
	pre-commit run --all-files

lint: ## check style with flake8
	flake8 process_improve tests

test: ## run tests quickly with the default Python
	clear
	rm -fr build/
	rm -fr dist/
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	python -W ignore -m pytest --exitfirst -v --new-first -r=s -r=a -n auto --cov=. #--pdb

# TODO: single test: ['-p', 'vscode_pytest', '--rootdir=/rootdir', '--capture=no', '/rootdir/tests/test_file.py::test_func']

coverage: ## check code coverage quickly with the default Python
	coverage html --precision=1 --skip-covered --skip-empty --title="Process Improve Coverage Report"
	coverage report --precision=1 --skip-covered --skip-empty
	python -m http.server 8080 --directory htmlcov

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/process_improve.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ process_improve
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	python -m http.server 8080 --directory docs/_build/html/


servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: clean check test  ## release to PyPI
	git pull
	uv build
	uv publish

install: clean ## install the package to the active Python's site-packages
	python setup.py install
