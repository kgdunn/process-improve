# CLAUDE.md — Project Context for AI Agents

## Overview

`process-improve` is a Python package for process improvement using data. It accompanies the online textbook [Process Improvement using Data](https://learnche.org/pid). The package provides multivariate analysis (PCA, PLS, TPLS), designed experiments, process monitoring, batch data analysis, and visualization tools.

**Repository:** https://github.com/kgdunn/process_improve
**License:** MIT
**Python:** >= 3.10 (CI runs on 3.11)

## Package Structure

```
process_improve/
    multivariate/    # PCA, PLS, TPLS, MCUVScaler, center, scale
    univariate/      # t_value, outlier_detection_multiple, confidence_interval
    experiments/     # Factorial designs (full, fractional, response surface)
    monitoring/      # Control charts (Shewhart, CUSUM, EWMA)
    batch/           # Batch process data alignment, features, preprocessing
    regression/      # Robust regression (repeated median, Theil-Sen)
    bivariate/       # Elbow detection, peak finding, area under curve
    visualization/   # Plotting utilities (raincloud plots, etc.)
    datasets/        # Sample datasets for examples and tests
```

## Versioning

The version is defined in `pyproject.toml` under `[project] version`. It uses 3-part semver: `MAJOR.MINOR.PATCH` (e.g., `1.3.1`).

**Auto-bump the version with every PR that changes code or configuration:**
- **PATCH** (last position, e.g., 1.3.0 → 1.3.1): bug fixes, CI/workflow changes, docs updates, dependency bumps, small refactors, and other minor changes.
- **MINOR** (middle position, e.g., 1.3.1 → 1.4.0): new features, new modules, significant API additions, or meaningful behavioral changes. Resets PATCH to 0.
- **If unsure** whether a change is major or minor, **ask the user** before bumping.

The PyPI publish workflow (`.github/workflows/publish.yml`) automatically detects version changes on push to `main` and publishes to PyPI when the version differs from the previous commit.

## Key Architectural Decisions

### sklearn API Compatibility
- **PCA** inherits from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`
- **PLS** inherits from `sklearn.cross_decomposition.PLSRegression`
- Fitted attributes use trailing `_` convention (e.g., `scores_`, `loadings_`, `spe_`, `hotellings_t2_`)
- `predict()` returns `sklearn.utils.Bunch` with named fields (not custom classes)
- `score()` follows sklearn convention (higher is better)
- `fit()` returns `self`

### PCA Fitted Attributes
`scores_`, `loadings_`, `spe_`, `hotellings_t2_`, `explained_variance_`, `r2_cumulative_`, `r2_per_component_`, `r2_per_variable_`, `scaling_factor_for_scores_`, `fitting_info_`, `has_missing_data_`

### PLS Fitted Attributes
`scores_` (X scores), `y_scores_`, `x_loadings_`, `y_loadings_`, `x_weights_`, `y_weights_`, `direct_weights_`, `beta_coefficients_`, `predictions_`, `spe_`, `hotellings_t2_`, `explained_variance_`, `r2_cumulative_`, `r2_per_component_`, `r2_per_variable_`, `r2y_per_variable_`, `rmse_`, `scaling_factor_for_scores_`, `fitting_info_`, `has_missing_data_`

### Convenience Method Binding
After `fit()`, PCA and PLS bind plot and limit methods as `functools.partial`:
```python
self.spe_limit = partial(spe_limit, model=self)
self.hotellings_t2_limit = partial(hotellings_t2_limit, ...)
self.score_plot = partial(score_plot, model=self)
self.spe_plot = partial(spe_plot, model=self)
self.t2_plot = partial(t2_plot, model=self)
self.loading_plot = partial(loading_plot, model=self)
```

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
- Linter: ruff (with `select = ["ALL"]` and specific ignores — see `pyproject.toml`)
- Formatter: black
- Type checking: mypy

## Testing

### Running Tests
```bash
# All multivariate tests
pytest tests/test_multivariate.py -v -o "addopts="

# PLS tests only
pytest tests/test_multivariate.py -v -k "pls" -o "addopts="

# PCA tests only
pytest tests/test_multivariate.py -v -k "pca" -o "addopts="

# Full suite
pytest -o "addopts="
```

### Test Conventions
- Use **real datasets** (LDPE, SIMCA) alongside synthetic data — do not remove real dataset tests
- Scale with `MCUVScaler().fit_transform(X)` in tests (not just `center()`)
- For synthetic PLS data, use `X.values @ beta` (not `X @ beta`) to avoid pandas column mismatch producing NaN
- Test fixtures load CSV data from `process_improve/datasets/multivariate/`
- New methods should have tests for both basic functionality and edge cases

## CI/CD

- **GitHub Actions**: `.github/workflows/pythonpackage.yml`
- **Python 3.11**, actions/checkout@v4, actions/setup-python@v5
- Install: `pip install -e ".[dev]"`
- Lint: `ruff check .`
- Test: `pytest`

## Documentation

- **System:** Sphinx with PyData theme
- **Build:** `cd docs && make html`
- **Docstring style:** NumPy (parsed by `sphinx.ext.napoleon`)

## Adding New Methods to PCA/PLS

1. Add the method to the class in `process_improve/multivariate/methods.py`
2. Use NumPy-style docstring with Parameters, Returns, Examples sections
3. Add tests in `tests/test_multivariate.py` using both real datasets and synthetic data
4. If the method needs to be on both PCA and PLS, implement on both with the same API signature
5. Run `pytest tests/test_multivariate.py -v -o "addopts="` to verify
