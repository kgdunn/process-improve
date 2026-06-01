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

### Test tiers

Tests are categorised via pytest markers so contributors can skip the ones
that are slow or network-dependent during local iteration. The full suite
runs in CI; the markers exist so a contributor doesn't pay for
multi-second or network-bound tests on every save.

| Marker | Meaning | Example | When to add |
|--------|---------|---------|-------------|
| (none) / `unit` | Fast in-process unit test. The implicit default; no decorator needed. | `test_check_random_state_resolves_int` | Default. |
| `integration` | Crosses module boundaries or exercises an external library's surface (statsmodels, plotly). | `test_pca_round_trip_via_sklearn_pipeline` | Add when the test wires several modules together. |
| `slow` | Multi-second wall-clock time, even on a fast laptop. | `test_tpls_full_fit_on_simca_dataset` | Add when the test takes >= 2 s. |
| `dataset` | Loads a bundled or remote real-world dataset (network or large file). | `test_oildoe_loads` | Add when the test depends on a real-data fixture, especially remote ones. |

Common opt-in / opt-out invocations:

```bash
# Skip network-dependent dataset tests (useful when offline):
pytest -m "not dataset" -o "addopts="

# Skip slow tests during local iteration:
pytest -m "not slow" -o "addopts="

# Run only the network-dependent dataset tests:
pytest -m dataset -o "addopts="

# CI runs the full suite (no marker filter).
```

The markers are registered in `pytest.ini`; pytest will warn if you
typo a marker name. Tag a test by adding the decorator above the
function (or above the `class` to tag the whole class):

```python
import pytest

@pytest.mark.dataset
def test_loads_real_data():
    ...
```

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
  or configuration (see "Versioning policy" below).
- Bugs and feature requests can also be filed on the
  [issue tracker](https://github.com/kgdunn/process-improve/issues).

## Versioning policy

The package uses three-part semantic versioning: `MAJOR.MINOR.PATCH`.

- **PATCH** (`1.22.8 -> 1.22.9`): bug fixes, CI / workflow changes,
  documentation updates, dependency bumps, small refactors. Public
  signatures are unchanged. Behaviour changes are only allowed when
  fixing a documented bug.
- **MINOR** (`1.22.9 -> 1.23.0`): new features, new modules,
  significant API additions, or meaningful behavioural changes.
  Backwards-compatible by default; deprecations follow the schedule
  in [`docs/development/deprecation_policy.rst`](docs/development/deprecation_policy.rst).
- **MAJOR** (`1.x.x -> 2.0.0`): incompatible removals. Anything
  previously deprecated under the policy can be removed here.

Reset rules:

- Bumping MINOR resets PATCH to 0 (`1.22.9 -> 1.23.0`).
- Bumping MAJOR resets MINOR and PATCH to 0 (`1.23.4 -> 2.0.0`).

If you are unsure whether a change is a PATCH or a MINOR, ask in the
PR; the maintainer decides on the boundary case.

### What counts as a breaking change

A change is breaking if any of these are true:

- A documented public name (function, class, attribute, kwarg) is
  removed, renamed, or moved.
- A public function's signature changes in a way that fails
  existing valid call sites (a positional becomes keyword-only,
  a required kwarg is added without a default, a kwarg default
  changes meaning).
- A `Bunch` return type loses or renames a field.
- A `@tool_spec` tool is renamed, has a schema field removed, or
  changes the meaning of an existing field.
- A documented behaviour (e.g. "PCA returns sign-flipped loadings
  by the Wold convention") is silently changed.

A change is **not** breaking if:

- A new optional kwarg is added with a default that preserves the
  previous behaviour.
- A new public function or attribute is added.
- A `Bunch` return type gains a field.
- A documented bug is fixed.
- Internal (`_`-prefixed) code is refactored without affecting any
  public surface.

Breaking changes always require a deprecation cycle; see the
linked policy below.

## Performance-regression policy

Public-API algorithms (PCA fit, PLS fit/predict, ``analyze_experiment``,
``evaluate_design``, batch alignment) are expected not to regress in
wall-clock time. The rule of thumb:

- A PR that knowingly slows a hot path by more than ~10% must say
  so in the PR description and justify the trade-off (correctness
  fix, security guard, deprecation prep).
- Unknowing regressions are caught by the perf-baseline CI job
  (planned: [ENG-15](https://github.com/kgdunn/process-improve/issues/297)).
  Until that job lands, the maintainer eyeballs perf on the hot
  paths during review.
- Internal refactors (private modules, internal classes) carry no
  perf SLA.

If a perf regression is unavoidable, the CHANGELOG entry says so
under "Changed".

## Policies referenced from this guide

These four documents pin the project's contributor-facing policies.
They are short, opinionated, and review feedback will reference
them directly:

- [Error-handling style](docs/development/error_handling.rst)
  ([ENG-11](https://github.com/kgdunn/process-improve/issues/293)) -- when to
  raise, when to assert, when to warn, when to use a dedicated
  exception class.
- [Reproducibility contract](docs/development/reproducibility.rst)
  ([ENG-08](https://github.com/kgdunn/process-improve/issues/290)) --
  ``random_state`` handling for every public function that
  touches an RNG.
- [Deprecation policy](docs/development/deprecation_policy.rst)
  ([ENG-22](https://github.com/kgdunn/process-improve/issues/304)) --
  the schedule for retiring public API, including the
  ``DeprecationWarning`` message format.
- This guide, the "Versioning policy" and "Performance-regression
  policy" sections above
  ([ENG-28](https://github.com/kgdunn/process-improve/issues/310)).
