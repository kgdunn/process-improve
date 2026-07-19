# CLAUDE.md - Project Context for AI Agents

## Overview

`process-improve` is a Python package for process improvement using data. It accompanies the online textbook [Process Improvement using Data](https://learnche.org/pid). The package provides multivariate analysis (PCA, PLS, and TPLS - *PLS for T-shaped data structures*, not "Total PLS" or "Three-way PLS"), designed experiments, process monitoring, batch data analysis, and visualization tools.

**Repository:** https://github.com/kgdunn/process-improve
**License:** MIT
**Python:** >= 3.10 (CI tests 3.10-3.13; 3.13 is the primary version for lint, typecheck, and coverage)

## Package Structure

The package uses a src layout: all code lives under `src/process_improve/`.

```
src/process_improve/
    multivariate/    # PCA, PLS, TPLS, MBPLS, MBPCA, MCUVScaler, center, scale
    univariate/      # t_value, outlier_detection_multiple, confidence_interval
    experiments/     # Factorial designs (full, fractional, response surface, optimal, OMARS)
    monitoring/      # Control charts (Shewhart, CUSUM, EWMA)
    batch/           # Batch process data alignment, features, preprocessing
    regression/      # Robust regression (repeated median, Theil-Sen)
    bivariate/       # Elbow detection, peak finding, area under curve
    sensory/         # Descriptive panel-data analysis (validate, panel check, relate)
    simulation/      # Fake-data / process-simulator subpackage
    visualization/   # Plotting utilities (raincloud plots, chart spec/IR, adapters)
    datasets/        # Sample datasets for examples and tests
    recipes.py       # Reusable analysis-recipe framework
    tool_spec.py     # @tool_spec decorator, tool registry, Anthropic tool-use specs
    tool_safety.py   # Subprocess isolation for tool calls over untrusted transports
    mcp_server.py    # FastMCP server exposing the registered tools (mcp extra)
    config.py        # Settings singleton reading PROCESS_IMPROVE_* env vars
    _extras.py       # Clean ImportError messages for missing optional extras
    _linalg.py       # Shared numerical-linear-algebra guards
    _random.py       # Shared random_state resolver
```

## Versioning

The version is defined in `pyproject.toml` under `[project] version`. It uses 3-part semver: `MAJOR.MINOR.PATCH` (e.g., `1.52.2`).

**Auto-bump the version with every PR that changes code or configuration:**
- **PATCH** (last position, e.g., 1.52.2 → 1.52.3): bug fixes, CI/workflow changes, docs updates, dependency bumps, small refactors, and other minor changes.
- **MINOR** (middle position, e.g., 1.52.3 → 1.53.0): new features, new modules, significant API additions, or meaningful behavioral changes. Resets PATCH to 0.
- **If unsure** whether a change is major or minor, **ask the user** before bumping.

**Keep `CITATION.cff` in sync with the version.** Whenever you bump the `version` in `pyproject.toml`, in the *same commit* set the `version:` field in `CITATION.cff` (repo root) to the identical value, and update `date-released:` to the current date. `pyproject.toml` and `CITATION.cff` must never report different versions.

The PyPI publish workflow (`.github/workflows/publish.yml`) is **manually gated** as of ENG-21 (#303). It runs only when a tag matching `v*` is pushed, or when a maintainer triggers `workflow_dispatch` from the Actions tab. Every release carries a sigstore attestation (PEP 740) and a CycloneDX SBOM attached to the GitHub release page; release notes are pulled from the matching `CHANGELOG.md` section. Bumping the version in a PR no longer publishes by itself; the maintainer must push the tag (or click Run workflow) once the PR has merged.

## Changelog

`CHANGELOG.md` (repo root) follows the [Keep a Changelog](https://keepachangelog.com) format.

For every PR or set of changes, **prompt the user to confirm whether a changelog entry is required.** User-facing changes (new features, API changes, bug fixes, behavioural changes) generally need one; internal-only changes (refactors, CI tweaks, edits to this file) generally do not. If an entry is required, write a relevant line under the appropriate version heading in `CHANGELOG.md` and stage it as part of the same commit.

New changelog lines go under the `## [Unreleased]` heading. When you bump the version in `pyproject.toml`, also update `CHANGELOG.md` in the same commit: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (today's date), add a fresh empty `## [Unreleased]` heading above it, and update the link-reference footer at the bottom of the file (the `[Unreleased]` compare link and a new `[X.Y.Z]` compare link).

## Key Architectural Decisions

### sklearn API Compatibility
- **PCA** inherits from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`
- **PLS** inherits from `sklearn.base.BaseEstimator`, `TransformerMixin`, and `RegressorMixin`.
  It deliberately does **not** inherit `sklearn.cross_decomposition.PLSRegression` (ENG-07 / #289):
  the estimators keep the lightweight sklearn *mixins* for API compatibility (`get_params`/`set_params`,
  `clone`, Pipeline support) but never couple to a concrete sklearn estimator's private attribute
  layout. The same applies to TPLS / MBPLS / MBPCA. `do not inherit from sklearn` here means the
  concrete estimators, not the mixins.
- Fitted attributes use trailing `_` convention (e.g., `scores_`, `loadings_`, `spe_`, `hotellings_t2_`)
  and are set only in `fit()`, never in `__init__` (sklearn requires `__init__` to set only the
  constructor parameters).
- `predict()` returns `sklearn.utils.Bunch` with named fields (not custom classes)
- `score()` follows sklearn convention (higher is better)
- `fit()` returns `self`

### PCA Fitted Attributes
`scores_`, `loadings_`, `spe_`, `hotellings_t2_`, `explained_variance_`, `r2_cumulative_`, `r2_per_component_`, `r2_per_variable_`, `scaling_factor_for_scores_`, `fitting_info_`, `has_missing_data_`

### PLS Fitted Attributes
`scores_` (X scores), `y_scores_`, `x_loadings_`, `y_loadings_`, `x_weights_`, `y_weights_`, `direct_weights_`, `beta_coefficients_`, `predictions_`, `spe_`, `hotellings_t2_`, `explained_variance_`, `r2_cumulative_`, `r2_per_component_`, `r2_per_variable_`, `r2y_per_variable_`, `rmse_`, `scaling_factor_for_scores_`, `fitting_info_`, `has_missing_data_`

### Convenience Method Binding
PCA / PLS / TPLS / MBPLS / MBPCA expose plot, limit, and diagnostic convenience methods
(`score_plot`, `spe_plot`, `loading_plot`, `spe_limit`, `score_limit`, `vip`, `eigenvalue_summary`,
`hotellings_t2_limit`, `ellipse_coordinates`, ...) that forward to the standalone functions in
`plots.py` / `_limits.py` / `_diagnostics.py`. As of ENG-05 (#287) these are **real methods defined
on the class**, not `functools.partial` instances bound in `fit()`:
```python
from process_improve.multivariate._common import _model_method

# Uniform `model=self` forwarders (defined at class-body time):
score_plot = _model_method(_score_plot)
spe_limit = _model_method(_spe_limit)
vip = _model_method(_vip)
# Methods that need fitted state are written out explicitly:
def hotellings_t2_limit(self, conf_level: float = 0.95) -> float:
    return _hotellings_t2_limit(conf_level=conf_level, n_components=self.n_components, n_rows=self.n_samples_)
```
This keeps `help` / `inspect.signature` accurate (they report the underlying function, minus `model`),
the fitted model picklable, and the methods overridable by subclasses. The standalone functions remain
importable for advanced callers. (TPLS's `spe_limit` is a separate nested dict-of-callables API and is
intentionally not a method.)

### Migration Helpers
Both PCA and PLS have `__getattr__` methods that raise `AttributeError` with helpful rename messages when old attribute names are used (e.g., `model.x_scores` tells you to use `model.scores_`).

## Coding Conventions

### Docstrings
- **NumPy style** throughout (Parameters, Returns, Examples, See Also, References sections)
- Use `----------` underlines for section headers
- Include type annotations in both signatures and docstrings

### Scaling
- Use `MCUVScaler` (mean-center, unit-variance) for scaling data before fitting models
- `center()` and `scale()` standalone functions also available but `MCUVScaler` is preferred (sklearn-compatible transformer)

### Variable Naming
- `N` = number of samples, `K` = number of features, `M` = number of targets, `A` = number of components
- These are used as local variables inside methods; stored as `n_samples_`, `n_features_in_`, `n_targets_`

### Code Quality
- Line length: 120 characters
- Linter: ruff (with `select = ["ALL"]` and specific ignores - see `pyproject.toml`)
- Formatter: ruff-format (do not add black, flake8, or isort config; ruff covers all of them)
- Type checking: mypy (CI gate covers `src/process_improve`)

### Prose style
- Do not use em-dashes (Unicode U+2014) in docs, docstrings, comments, commit
  messages, or PR descriptions. Use a hyphen (`-`), a semicolon, or split the
  sentence instead. This applies to Markdown, reStructuredText, Python
  docstrings/comments, and YAML prose content.

## Testing

### Running Tests

Defaults in `pytest.ini` include xdist parallelism and the coverage gate; a
plain `pytest` works. Debug helpers (`--pdb`, `-x`, `-v`) are not defaults;
pass them manually when needed.

```bash
# Full suite (parallel, with the coverage gate)
uv run pytest

# One file (multivariate tests are split across ~15 files:
# tests/test_multivariate.py, test_multiblock_reference.py,
# test_multivariate_tsr.py, test_multivariate_robustness.py, ...)
uv run pytest tests/test_multivariate.py --no-cov

# By keyword
uv run pytest -k "pls" --no-cov
```

### Test Conventions
- Use **real datasets** (LDPE, SIMCA) alongside synthetic data - do not remove real dataset tests
- Scale with `MCUVScaler().fit_transform(X)` in tests (not just `center()`)
- For synthetic PLS data, use `X.values @ beta` (not `X @ beta`) to avoid pandas column mismatch producing NaN
- Test fixtures load CSV data from `src/process_improve/datasets/multivariate/`
- New methods should have tests for both basic functionality and edge cases
- Guard optional dependencies with `pytest.importorskip` (and probe binaries that
  can be present but non-executable, e.g. pulp's bundled CBC solver)

## CI/CD

Workflows in `.github/workflows/`:

- **run-tests.yml**: `lint` (`uv run ruff check .`), `typecheck` (blocking
  `uv run mypy src/process_improve`), `test` (pytest matrix over Python
  3.10-3.13 and ubuntu/windows/macos), and `test-under-dash-O` (runs the suite
  under `python -O` to catch load-bearing asserts). All jobs install with
  `uv sync --dev --all-extras`.
- **docs.yml**: strict Sphinx build (`-W`, notebooks executed) and GitHub
  Pages deploy on main.
- **publish.yml**: tag-gated PyPI publish (see Versioning above).
- **codeql.yml**: weekly and per-PR security scanning.

## Documentation

- **System:** Sphinx with PyData theme
- **Build:** `cd docs && make html`
- **Docstring style:** NumPy (parsed by `sphinx.ext.napoleon`)

## Adding New Methods to PCA/PLS

1. Add the method to the class: PCA lives in `src/process_improve/multivariate/_pca.py`,
   PLS in `_pls.py` (shared plotting in `plots.py`, limits in `_limits.py`,
   diagnostics in `_diagnostics.py`; `methods.py` is only a re-export shim)
2. Use NumPy-style docstring with Parameters, Returns, Examples sections
3. Add tests in `tests/test_multivariate.py` (or the more specific
   `tests/test_multivariate_*.py` file) using both real datasets and synthetic data
4. If the method needs to be on both PCA and PLS, implement on both with the same API signature
5. Run `uv run pytest tests/test_multivariate.py --no-cov` to verify

## Git & PR workflow (for Claude Code sessions)

**Default workflow for any non-trivial task:**

1. **Open a blank PR up front**, before doing the work. Push an empty commit
   (or the first micro-commit) to `claude/<task-slug>` and open the PR as
   ready-for-review with a description of what's about to be done. Do not
   open it as a draft.
2. **Micro-commit as you go.** Each commit should be a small, self-contained
   step (one module's tests, one config tweak, one bug fix) - not a single
   end-of-session megacommit.
3. **Push regularly.** After every micro-commit (or at worst every 2-3),
   `git push` to the same branch so the PR reflects current progress and the
   user can watch it land.

This is the default - don't ask the user whether to do it, just do it.

**Never push lock files.** Claude Code sessions must not stage, commit, or push
any dependency lock files. Lock-file updates are performed manually by the
repository owner.

Specifically, do **not** include the following in any commit or PR opened from
a Claude Code session:

- `uv.lock`
- `poetry.lock`
- `Pipfile.lock`
- `requirements.lock` / pip-tools compiled lockfiles
- any equivalent regenerated lock artifact

If a command (e.g. `uv sync`, `pip install`) regenerates a lock file during a
session, leave the file uncommitted. If it has already been staged, unstage it
(`git restore --staged <lockfile>`) before committing. The user will refresh
lock files manually outside of Claude Code sessions.

## Updating this file (CLAUDE.md)

If during a session you notice a recurring pattern, convention, or piece of
project context that you think belongs in `CLAUDE.md`, **do not add it
yourself**. Surface the proposed addition in chat - the wording you would
add, where you would put it, and why you think it is reusable - and ask the
user whether it should be recorded here. The user decides what gets
canonised in this file.
