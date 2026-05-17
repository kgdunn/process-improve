# Contributing to process-improve

Bug reports, feature requests, and pull requests are welcome. This guide
covers how to set up a development environment and the conventions the
project follows.

## Development setup

Clone the repository and install it in editable mode with the development
extras:

```bash
git clone https://github.com/kgdunn/process-improve.git
cd process-improve
pip install -e ".[dev]"
```

This brings in the test runner, linter, formatter, type checker, and the
documentation toolchain.

## Running tests

The full suite:

```bash
pytest -o "addopts="
```

A focused subset, for example the multivariate tests:

```bash
pytest tests/test_multivariate.py -v -o "addopts="
```

Tests use real datasets (LDPE, SIMCA) alongside synthetic data; please keep
the real-dataset tests in place when adding new ones. New methods should have
tests for both basic functionality and edge cases.

## Linting and formatting

```bash
ruff check .          # lint
black .               # format
mypy process_improve  # type check
```

The line length is 120 characters.

## Code style

- **Docstrings:** NumPy style throughout (Parameters, Returns, Examples,
  See Also, References), with type annotations in both signatures and
  docstrings.
- **Fitted attributes:** follow the sklearn convention of a trailing
  underscore (`scores_`, `loadings_`, `spe_`, ...).
- **Scaling:** prefer `MCUVScaler` (mean-center, unit-variance) for preparing
  data before fitting models.
- **Prose:** do not use em-dashes in code, comments, docstrings, or commit
  messages; use a hyphen, a semicolon, or split the sentence.

## Adding new methods to PCA / PLS

1. Add the method to the class in `process_improve/multivariate/methods.py`.
2. Use a NumPy-style docstring with Parameters, Returns, and Examples
   sections.
3. Add tests in `tests/test_multivariate.py` using both real datasets and
   synthetic data.
4. If the method belongs on both PCA and PLS, implement it on both with the
   same API signature.
5. Verify with `pytest tests/test_multivariate.py -v -o "addopts="`.

## Submitting changes

- Open a pull request against `main` with a clear description of the change
  and why it is needed.
- The version in `pyproject.toml` is bumped with every PR that changes code
  or configuration (PATCH for fixes and small changes, MINOR for new
  features).
- Bugs and feature requests can also be filed on the
  [issue tracker](https://github.com/kgdunn/process-improve/issues).
