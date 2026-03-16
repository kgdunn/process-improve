.PHONY: clean setup docs servedocs lint test coverage check release install help
.DEFAULT_GOAL := help

help:		## Help in the makefile
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

clean:		## Remove build artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr .ruff_cache/
	rm -fr .mypy_cache/
	rm -fr docs/_build/

setup: clean	## Clean, then set up a fresh dev environment
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --dev
	uv run pre-commit install

check:		## if the first command gives a return, then stage those files, then run pre-commit
	git update-index --refresh
	uv run pre-commit run --all-files

lint:		## check style with ruff
	uv run ruff check .
	uv run ruff format --check .

test:		## run tests quickly with the default Python
	clear
	rm -fr build/
	rm -fr dist/
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	uv run python -W ignore -m pytest -v  -r=s -r=a -n auto --cov=.

# TODO: single test: ['-p', 'vscode_pytest', '--rootdir=/rootdir', '--capture=no', '/rootdir/tests/test_file.py::test_func']

coverage:	## check code coverage quickly with the default Python
	coverage html --precision=1 --skip-covered --skip-empty --title="Process Improve Coverage Report"
	coverage report --precision=1 --skip-covered --skip-empty
	uv run python -m http.server 8080 --directory htmlcov

docs:		## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	uv run python -m http.server 8080 --directory docs/_build/html/


servedocs: docs 	## compile the docs watching for changes
	uv run watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: check test  	## release to PyPI
	git pull
	uv build
	uv publish

install: 	## install the package in dev mode
	uv sync --dev
